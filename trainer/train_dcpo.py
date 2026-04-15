import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from collections import defaultdict
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (Logger, is_main_process, lm_checkpoint, safe_save,
                                   init_distributed_mode, setup_seed, SkipBatchSampler,
                                   init_model, LMForRewardModel)
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps
from trainer.shared_rl_utils import (
    calculate_rewards, compute_completion_mask,
    compute_kl_terms, ratio_stats, clipped_fractions,
    clip_and_get_grad_norm, MovingAverage, build_log_dict,
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# DCPO: Dynamic Clipping Policy Optimization
# arXiv: 2509.02333
#
# Clipping mechanism: TOKEN-PROBABILITY-ADAPTIVE dynamic bounds
#
# GRPO's fixed [1-ε, 1+ε] clip interval is the same for every token regardless
# of how likely that token was under π_old.  Tokens with very small old prob
# (q → 0) can have ratios that stray far from 1 even for small absolute changes
# in log-prob — yet they're still clipped at the same threshold as common tokens.
#
# DCPO uses bounds that adapt to the token's old probability q = exp(old_logp):
#
#   lower(q) = 0.5 + 0.5 · sqrt( max(1 − 4·ε_low / q, 0) )
#   upper(q) = 0.5 + 0.5 · sqrt(1 + 4·ε_high / q)
#   (both capped at MAX_CLIP = 10 for numerical safety)
#
# Effect:
#   • For high-prob tokens (q → 1): bounds ≈ [1-ε_low, 1+ε_high] (like GRPO)
#   • For low-prob tokens  (q → 0): bounds widen dramatically
#     → exploration tokens get a much wider clip window
#     → ~10x fewer zero-gradient tokens compared to GRPO (per the paper)
#
# Smooth Advantage Standardisation (SAS):
#   DCPO also replaces standard group-relative advantage normalisation with a
#   blend of step-local and per-prompt cumulative (Welford) statistics.
#   [DIFFER] This maintains running state in `prompt_stats`, which is unique
#   to DCPO.  All other algorithms use the stateless compute_group_advantages().
# ─────────────────────────────────────────────────────────────────────────────

MAX_CLIP = 10.0  # safety cap on dynamic clip bounds (prevents overtraining on rare tokens)


def dcpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model,
                     prompt_stats, global_step_ref, reward_ma, kl_ma, start_step=0, wandb=None):
    """
    prompt_stats:      dict[str → {'n', 'mean', 'M2'}]  — Welford stats per prompt (SAS)
    global_step_ref:   list of length 1  — mutable global step counter for SAS blend weight
    """
    grad_norm_val = float('nan')

    for step, batch in enumerate(loader, start=start_step + 1):
        global_step_ref[0] += 1
        gs = global_step_ref[0]

        prompts = batch['prompt']
        prompt_inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            return_token_type_ids=False, padding_side="left", add_special_tokens=False
        ).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"]      = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ── Rollout ──────────────────────────────────────────────────────────
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs             = rollout_result.output_ids
        completion_ids      = rollout_result.completion_ids
        completions         = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()

        # ── Policy forward pass ───────────────────────────────────────────────
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res      = model_unwrapped(outputs)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits   = res.logits[:, :-1, :]
            per_token_logps = (F.log_softmax(logits, dim=-1)
                               .gather(2, outputs[:, 1:].unsqueeze(-1))
                               .squeeze(-1)[:, -completion_ids.size(1):])

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))

        # ── Shared reward + masking + KL ──────────────────────────────────────
        rewards         = calculate_rewards(prompts, completions, args.num_generations,
                                            args.device, reward_model)
        completion_mask = compute_completion_mask(completion_ids, tokenizer.eos_token_id)
        per_token_kl, kl_ref_val = compute_kl_terms(
            ref_per_token_logps, per_token_logps, completion_mask)

        # ── DCPO: Smooth Advantage Standardisation (SAS) ─────────────────────
        # [DIFFER] Replaces stateless compute_group_advantages() with a blend
        # of step-local and per-prompt cumulative (Welford) normalisation.
        # Motivation: eliminates zero-advantage batches without Dynamic Sampling.
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, G]
        mean_new = grouped_rewards.mean(dim=1)
        std_new  = grouped_rewards.std(dim=1).clamp(min=1e-4)

        cum_mean = torch.zeros(len(prompts), device=args.device)
        cum_std  = torch.ones(len(prompts), device=args.device)
        for i, p in enumerate(prompts):
            s = prompt_stats[p]
            for r in grouped_rewards[i].tolist():
                s['n']    += 1
                delta      = r - s['mean']
                s['mean'] += delta / s['n']
                s['M2']   += delta * (r - s['mean'])
            cum_mean[i] = s['mean']
            cum_std[i]  = max((s['M2'] / max(s['n'] - 1, 1)) ** 0.5, 1e-4)

        # Blend weights: early steps rely more on cumulative, later more on current
        w_new = (gs - 1) / gs
        w_cum = 1.0 / gs

        adv_new     = (grouped_rewards - mean_new.unsqueeze(1)) / std_new.unsqueeze(1)
        adv_cum     = (grouped_rewards - cum_mean.unsqueeze(1)) / cum_std.unsqueeze(1)
        sa_new      = w_new * adv_new + w_cum * adv_cum
        sa_cum      = w_cum * adv_new + w_new * adv_cum
        advantages  = torch.where(sa_new.abs() < sa_cum.abs(), sa_new, sa_cum).reshape(-1)  # [B*G]

        # ── DCPO: Dynamic-Adaptive Clipping Bounds ────────────────────────────
        # [DIFFER] Clip bounds vary per-token based on old probability q.
        # Low-prob tokens get wider bounds → fewer zero-gradient tokens.
        old_prob    = old_per_token_logps.exp().clamp(1e-6, 1.0)              # q [B*G, L]

        lower_bound = (0.5 + 0.5 * torch.sqrt(
            torch.clamp(1.0 - 4.0 * args.epsilon_low  / old_prob, min=0.0)
        )).clamp(max=MAX_CLIP)

        upper_bound = (0.5 + 0.5 * torch.sqrt(
            1.0 + 4.0 * args.epsilon_high / old_prob
        )).clamp(max=MAX_CLIP)

        ratio         = torch.exp(per_token_logps - old_per_token_logps)     # [B*G, L]
        adv           = advantages.unsqueeze(1)
        clipped_ratio = torch.max(torch.min(ratio, upper_bound), lower_bound)
        per_token_obj = torch.min(ratio * adv, clipped_ratio * adv)

        seq_loss    = -((per_token_obj - args.beta * per_token_kl) * completion_mask).sum(dim=1) \
                      / completion_mask.sum(dim=1).clamp(min=1)
        policy_loss = seq_loss.mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        # ── Gradient update ───────────────────────────────────────────────────
        if step % args.accumulation_steps == 0:
            grad_norm_val = clip_and_get_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if is_main_process() and step % args.save_interval == 0:
                rollout_engine.update_policy(model)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_interval == 0 or step == iters:
            avg_reward = rewards.mean().item()
            avg_len    = completion_mask.sum(dim=1).float().mean().item()
            policy_loss_val = loss.item() * args.accumulation_steps

            r_stats = ratio_stats(ratio, completion_mask)
            # For DCPO we also report the dynamic bound statistics
            m = completion_mask.bool()
            lower_mean = lower_bound[m].mean().item()
            upper_mean = upper_bound[m].mean().item()
            # Fraction where dynamic bounds differ from standard GRPO bounds
            # (i.e., tokens for which DAC actually widened the clip window)
            eps     = max(args.epsilon_low, args.epsilon_high)
            dyn_clipped = ((ratio < lower_bound) | (ratio > upper_bound)) & m
            std_clipped = ((ratio < (1 - eps)) | (ratio > (1 + eps))) & m
            total   = m.float().sum().item() + 1e-8
            dyn_frac     = dyn_clipped.float().sum().item() / total
            extra_tokens = (std_clipped.float().sum() - dyn_clipped.float().sum()).clamp(min=0).item() / total

            core = {
                "train/reward":           avg_reward,
                "train/policy_loss":      policy_loss_val,
                "train/kl_ref":           kl_ref_val,
                "train/avg_response_len": avg_len,
                "train/grad_norm":        grad_norm_val,
                "train/learning_rate":    optimizer.param_groups[0]['lr'],
                "train/advantages_mean":  advantages.mean().item(),
                "train/advantages_std":   advantages.std().item(),
            }
            clip_metrics = {
                **r_stats,
                "clip/lower_bound_mean":   lower_mean,
                "clip/upper_bound_mean":   upper_mean,
                "clip/dyn_clipped_fraction": dyn_frac,
                # extra_tokens = tokens clipped by GRPO that DCPO lets through
                # (positive value means DCPO reduced dead zones — key paper claim)
                "clip/clipped_fraction":   dyn_frac,  # for cross-algorithm comparability
            }
            stability = {
                "stability/reward_ma": reward_ma.update(avg_reward),
                "stability/kl_ma":     kl_ma.update(kl_ref_val),
            }

            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters})  '
                f'Reward:{avg_reward:.4f}  KL:{kl_ref_val:.4f}  '
                f'Loss:{policy_loss_val:.4f}  Len:{avg_len:.1f}  '
                f'DynClipped:{dyn_frac:.3f}  ExtraAlive:{extra_tokens:.3f}  '
                f'LoBound:{lower_mean:.4f}  HiBound:{upper_mean:.4f}  '
                f'RatioMean:{r_stats["clip/ratio_mean"]:.4f}  '
                f'GradNorm:{grad_norm_val:.3f}  LR:{optimizer.param_groups[0]["lr"]:.2e}'
            )

            if wandb and is_main_process():
                wandb.log(build_log_dict(core, clip_metrics, stability))

        # ── Checkpoint ───────────────────────────────────────────────────────
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model  = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model  = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            safe_save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                          scheduler=scheduler)
            model.train()
            del state_dict

    if step > start_step and step % args.accumulation_steps != 0:
        grad_norm_val = clip_and_get_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if is_main_process() and step % args.save_interval == 0:
            rollout_engine.update_policy(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCPO: Dynamic Clipping Policy Optimization")
    # ── Infrastructure (identical across ALL algorithms) ──────────────────────
    parser.add_argument("--save_dir",        type=str,   default="../out")
    parser.add_argument("--save_weight",     type=str,   default="dcpo")
    parser.add_argument("--epochs",          type=int,   default=1)
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--learning_rate",   type=float, default=3e-7)
    parser.add_argument("--device",          type=str,   default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",           type=str,   default="bfloat16")
    parser.add_argument("--num_workers",     type=int,   default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--log_interval",    type=int,   default=1)
    parser.add_argument("--save_interval",   type=int,   default=10)
    parser.add_argument("--hidden_size",     type=int,   default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe",         type=int,   default=0, choices=[0, 1])
    parser.add_argument("--max_seq_len",     type=int,   default=768)
    parser.add_argument("--max_gen_len",     type=int,   default=1024)
    parser.add_argument("--data_path",       type=str,   default="../dataset/rlaif.jsonl")
    parser.add_argument("--num_generations", type=int,   default=6)
    parser.add_argument("--beta",            type=float, default=0.1)
    parser.add_argument("--from_weight",     type=str,   default="full_sft")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    parser.add_argument("--from_resume",     type=int,   default=0, choices=[0, 1])
    parser.add_argument("--use_wandb",       action="store_true")
    parser.add_argument("--wandb_project",   type=str,   default="ClipComparison")
    parser.add_argument("--use_compile",     type=int,   default=0, choices=[0, 1])
    parser.add_argument("--debug_mode",      action="store_true")
    parser.add_argument("--debug_interval",  type=int,   default=20)
    parser.add_argument("--thinking_ratio",  type=float, default=0.9)
    # ── DCPO-specific: dynamic adaptive clipping ──────────────────────────────
    # [DIFFER] ε_low / ε_high are used in the DAC formula, not directly as clip bounds.
    # The actual bounds vary per-token and are wider than these values for low-prob tokens.
    parser.add_argument("--epsilon_low",  type=float, default=0.2,
                        help="Controls dynamic lower bound width. lower(q) = 0.5+0.5*sqrt(max(1-4ε/q,0))")
    parser.add_argument("--epsilon_high", type=float, default=0.2,
                        help="Controls dynamic upper bound width. upper(q) = 0.5+0.5*sqrt(1+4ε/q)")
    args = parser.parse_args()
    args.reward_model_path = os.path.abspath(args.reward_model_path)

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len,
                               use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        wandb.init(project=args.wandb_project,
                   name=f"DCPO-epsL{args.epsilon_low}-epsH{args.epsilon_high}-LR{args.learning_rate}",
                   id=wandb_id, resume='allow' if wandb_id else None,
                   config=vars(args))

    model,     tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ref_model, _         = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    reward_model   = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    rollout_engine = create_rollout_engine(
        engine_type="torch", policy_model=model, tokenizer=tokenizer,
        device=args.device, autocast_ctx=autocast_ctx)

    train_ds      = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len,
                                 thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer     = optim.AdamW(model.parameters(), lr=args.learning_rate)
    iters         = len(DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler))
    total_steps   = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler     = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step  = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    if is_main_process(): rollout_engine.update_policy(model)

    # Per-prompt Welford stats for SAS  [DIFFER: stateful across steps, unique to DCPO]
    prompt_stats    = defaultdict(lambda: {'n': 0, 'mean': 0.0, 'M2': 0.0})
    global_step_ref = [0]
    reward_ma       = MovingAverage(alpha=0.05)
    kl_ma           = MovingAverage(alpha=0.05)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip    = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        n_iters = len(loader) + skip if skip > 0 else len(loader)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: resuming from step {start_step+1}')
        dcpo_train_epoch(epoch, loader, n_iters, rollout_engine, ref_model, reward_model,
                         prompt_stats, global_step_ref, reward_ma, kl_ma, skip, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
