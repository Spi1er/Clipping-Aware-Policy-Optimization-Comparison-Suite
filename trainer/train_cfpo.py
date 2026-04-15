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
    calculate_rewards, compute_completion_mask, compute_group_advantages,
    compute_kl_terms, ratio_stats,
    clip_and_get_grad_norm, MovingAverage, build_log_dict,
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CFPO: Clipping-Free Policy Optimization
# arXiv: 2601.22801
#
# Clipping mechanism: NONE — replaced by a convex quadratic penalty
#
# Standard PPO/GRPO: L = min(r·Â, clip(r,1±ε)·Â)
#   Creates flat "dead zones" beyond [1-ε, 1+ε] where gradient = 0.
#
# CFPO: L = r·Â − |Â|/(2ε) · (r−1)²  − β·KL
#
# Properties:
#   • No zero-gradient regions: gradient = Â − |Â|/ε·(r−1)·r is non-zero everywhere.
#   • Convex trust region: quadratic term penalises deviation from r=1 with magnitude
#     proportional to |Â| (bigger advantages justify bigger updates).
#   • Smooth everywhere: no sharp clip boundary.
#   • Exact equivalence at r=1: reduces to REINFORCE at on-policy start.
#
# The "off_policy_fraction" metric is the fraction of tokens where |r-1|>ε —
# this is what GRPO would clip; for CFPO these tokens still contribute to the
# gradient (via the penalty term), so comparing this fraction across methods
# shows how much behaviour differs.
# ─────────────────────────────────────────────────────────────────────────────


def cfpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model,
                     reward_ma, kl_ma, start_step=0, wandb=None):
    grad_norm_val = float('nan')

    for step, batch in enumerate(loader, start=start_step + 1):
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

        # ── Shared: rewards, advantages, mask, KL ────────────────────────────
        rewards         = calculate_rewards(prompts, completions, args.num_generations,
                                            args.device, reward_model)
        advantages      = compute_group_advantages(rewards, args.num_generations)
        completion_mask = compute_completion_mask(completion_ids, tokenizer.eos_token_id)
        per_token_kl, kl_ref_val = compute_kl_terms(
            ref_per_token_logps, per_token_logps, completion_mask)

        # ── CFPO-specific: quadratic penalty (no clipping) ───────────────────
        # [DIFFER] The core algorithmic difference: no min(), no clip().
        # Gradient = (Â − |Â|/ε·(r−1)) · r · ∇log π_θ  — non-zero everywhere.
        ratio = torch.exp(per_token_logps - old_per_token_logps)             # [B*G, L]
        adv   = advantages.unsqueeze(1)                                       # [B*G, 1]

        # Quadratic trust-region penalty: |Â|/(2ε) · (r−1)²
        quadratic_penalty = adv.abs() / (2.0 * args.epsilon) * (ratio - 1.0) ** 2
        per_token_obj     = ratio * adv - quadratic_penalty                  # no hard clip
        per_token_loss    = -(per_token_obj - args.beta * per_token_kl)

        # [FAIR] Same per-sequence mean → batch mean aggregation as GRPO.
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1)
                       / completion_mask.sum(dim=1).clamp(min=1)).mean()
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

            # [DIFFER] CFPO-specific: no hard clip, so we report:
            # 1. off_policy_fraction: tokens where GRPO *would* have clipped (|r-1|>ε)
            #    — for CFPO these still get a gradient (via penalty), showing the key diff
            # 2. quadratic_penalty_mean: average magnitude of the penalty term
            m = completion_mask.bool()
            off_policy_frac = ((ratio - 1.0).abs() > args.epsilon)[m].float().mean().item()
            penalty_mean    = quadratic_penalty[m].mean().item()

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
                # These are the key CFPO-specific metrics:
                "clip/off_policy_fraction":    off_policy_frac,   # = GRPO's clipped_fraction equivalent
                "clip/quadratic_penalty_mean": penalty_mean,      # no analogues in hard-clip methods
                # For cross-algorithm comparability, also log NaN for hard-clip keys
                # (build_log_dict fills absent keys with NaN automatically)
            }
            stability = {
                "stability/reward_ma": reward_ma.update(avg_reward),
                "stability/kl_ma":     kl_ma.update(kl_ref_val),
            }

            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters})  '
                f'Reward:{avg_reward:.4f}  KL:{kl_ref_val:.4f}  '
                f'Loss:{policy_loss_val:.4f}  Len:{avg_len:.1f}  '
                f'OffPolicy:{off_policy_frac:.3f}  PenaltyMean:{penalty_mean:.4f}  '
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
    parser = argparse.ArgumentParser(description="CFPO: Clipping-Free Policy Optimization")
    # ── Infrastructure (identical across ALL algorithms) ──────────────────────
    parser.add_argument("--save_dir",        type=str,   default="../out")
    parser.add_argument("--save_weight",     type=str,   default="cfpo")
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
    # ── CFPO-specific: trust-region width ────────────────────────────────────
    # [DIFFER] ε here controls the quadratic penalty magnitude, not a hard clip bound.
    # It's conceptually equivalent to PPO's ε but mechanically different.
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Trust region width ε for the quadratic penalty: |A|/(2ε)·(r-1)². "
                             "Same value as GRPO's clip ε for fair comparison. "
                             "Larger ε = more permissive updates (weaker penalty).")
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
                   name=f"CFPO-eps{args.epsilon}-LR{args.learning_rate}",
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

    reward_ma = MovingAverage(alpha=0.05)
    kl_ma     = MovingAverage(alpha=0.05)

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
        cfpo_train_epoch(epoch, loader, n_iters, rollout_engine, ref_model, reward_model,
                         reward_ma, kl_ma, skip, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
