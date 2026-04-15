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
    compute_kl_terms, ratio_stats, seq_ratio_stats, clipped_fractions,
    clip_and_get_grad_norm, MovingAverage, build_log_dict,
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# GSPO: Group Sequence Policy Optimization
# arXiv: 2507.18071
#
# Clipping mechanism: SEQUENCE-LEVEL PPO min-clip
#
# Key insight: GRPO's token-level IS weights are each derived from a single
# sample of π_old(·|ctx), making them high-variance.  This variance accumulates
# over long sequences and is amplified by token-level clipping, causing
# catastrophic collapse in large models.
#
# GSPO replaces the per-token IS weight with a length-normalised sequence ratio:
#
#   s_i(θ) = exp( 1/|y_i| · Σ_t log[π_θ(y_i,t) / π_old(y_i,t)] )
#
# The 1/|y_i| exponent brings the ratio onto a consistent numerical scale
# regardless of response length.  Standard PPO min-clip is then applied
# at the SEQUENCE level:
#
#   J_GSPO = E[ min(s_i·Â_i,  clip(s_i, 1-ε, 1+ε)·Â_i) ]
#
# Critical: because s_i ≈ (token ratio)^(1/L), the sequence-level ratio
# lives MUCH closer to 1 than token-level ratios.  Using ε=0.2 (GRPO default)
# would make clipping effectively inactive.  GSPO requires a much smaller ε.
#
# GSPO ε vs GRPO ε:
#   Token-level ratio r_t = π_θ/π_old   ← varies widely per token
#   Sequence-level ratio s_i ≈ r̄^(1/L)  ← much closer to 1 for long sequences
#   ε must be scaled accordingly: ε_seq ≈ ε_token / L (rough approximation)
#   Paper (30B model): ε ≈ 3e-4 to 4e-4.  For small models: 0.001–0.01.
#   [DIFFER] This is a fundamental property of GSPO's algorithm, not a tuning choice.
# ─────────────────────────────────────────────────────────────────────────────


def gspo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model,
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

        # ── GSPO-specific: sequence-level importance ratio ────────────────────
        # [DIFFER] This replaces token-level ratios with a length-normalised sequence ratio.
        # Per-token log-ratio: log r_i,t = log π_θ − log π_old
        log_ratio_per_token = per_token_logps - old_per_token_logps             # [B*G, L]
        seq_len             = completion_mask.float().sum(dim=1).clamp(min=1)  # [B*G]

        # Length-normalised sequence log-ratio: 1/|y_i| · Σ_t mask_t · log_r_i,t
        seq_log_ratio = (log_ratio_per_token * completion_mask).sum(dim=1) / seq_len  # [B*G]

        # Sequence-level importance ratio s_i = exp(seq_log_ratio)
        # [DIFFER] GSPO applies PPO clip at this sequence level, NOT per token.
        seq_ratio       = seq_log_ratio.exp()                                   # [B*G]
        clipped_s       = torch.clamp(seq_ratio, 1.0 - args.epsilon, 1.0 + args.epsilon)
        seq_obj         = torch.min(seq_ratio * advantages, clipped_s * advantages)  # [B*G]

        # Token-level KL penalty averaged per sequence (for regularisation)
        seq_kl   = (per_token_kl * completion_mask).sum(dim=1) / seq_len
        seq_loss = -(seq_obj - args.beta * seq_kl)                              # [B*G]

        # [FAIR] Same loss aggregation scale: mean over sequences in the batch.
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
            avg_len    = seq_len.mean().item()
            policy_loss_val = loss.item() * args.accumulation_steps

            # GSPO: primary ratio stats are SEQUENCE-level (not token-level)
            # [DIFFER] seq_ratio lives near 1 with very tight spread vs token ratios
            s_stats      = seq_ratio_stats(seq_ratio)
            seq_clip_frac = (seq_ratio != clipped_s).float().mean().item()

            # Also compute token-level ratio stats for cross-algorithm comparison
            token_ratio  = torch.exp(log_ratio_per_token)
            t_stats      = ratio_stats(token_ratio, completion_mask)

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
                **s_stats,
                **t_stats,   # token-level for cross-algorithm comparability
                "clip/seq_clipped_fraction": seq_clip_frac,
            }
            stability = {
                "stability/reward_ma": reward_ma.update(avg_reward),
                "stability/kl_ma":     kl_ma.update(kl_ref_val),
            }

            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters})  '
                f'Reward:{avg_reward:.4f}  KL:{kl_ref_val:.4f}  '
                f'Loss:{policy_loss_val:.4f}  Len:{avg_len:.1f}  '
                f'SeqRatio:{s_stats["clip/seq_ratio_mean"]:.5f}±{s_stats["clip/seq_ratio_std"]:.5f}  '
                f'SeqClipped:{seq_clip_frac:.3f}  '
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
    parser = argparse.ArgumentParser(description="GSPO: Group Sequence Policy Optimization")
    # ── Infrastructure (identical across ALL algorithms) ──────────────────────
    parser.add_argument("--save_dir",        type=str,   default="../out")
    parser.add_argument("--save_weight",     type=str,   default="gspo")
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
    # ── GSPO-specific: sequence-level clipping ────────────────────────────────
    # [DIFFER] Must be MUCH smaller than token-level ε (GRPO uses 0.2, GSPO needs ≪0.1).
    # Reason: s_i = (π_θ/π_old)^(1/L) ≈ r̄^(1/L) lives near 1 with tight spread.
    # Using ε=0.2 here would make clipping essentially inactive for GSPO.
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="Sequence-level clip range for s_i=(π_θ/π_old)^(1/|y|). "
                             "Must be << token-level ε. Paper: 3e-4–4e-4 (30B model). "
                             "Recommended for small models: 0.001–0.01.")
    args = parser.parse_args()
    args.reward_model_path = os.path.abspath(args.reward_model_path)

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    if args.epsilon > 0.05 and is_main_process():
        Logger(
            f"[GSPO WARNING] --epsilon={args.epsilon} is likely too large for sequence-level clipping. "
            f"GSPO clips s_i=(π_θ/π_old)^(1/|y|); recommended range 0.001–0.01."
        )

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
                   name=f"GSPO-eps{args.epsilon}-LR{args.learning_rate}",
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
        gspo_train_epoch(epoch, loader, n_iters, rollout_engine, ref_model, reward_model,
                         reward_ma, kl_ma, skip, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
