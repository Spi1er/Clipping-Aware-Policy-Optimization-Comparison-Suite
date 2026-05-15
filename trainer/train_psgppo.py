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
                                   init_model, LMForRewardModel, init_wandb_safely)
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps
from trainer.shared_rl_utils import (
    calculate_rewards, compute_completion_mask, compute_group_advantages,
    ratio_stats, entropy_stats,
    clip_and_get_grad_norm, MovingAverage, build_log_dict,
)
from trainer.algorithms.psgppo_sequence import compute_sequence_psgppo_loss

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PS-GPPO: Per-Step Gradient-Preserving Policy Optimization
#
# Clipping mechanism: STEP-LEVEL geometric-mean ratio + gradient-preserving
#                     region classification (CE-GPPO style)
#
# Combines two building blocks:
#
#   1. GSPO's length-normalised step ratio:
#      s_k = exp( 1/L_k · Σ_t log[π_θ(y_t|ctx) / π_old(y_t|ctx)] )
#      In single-turn setting, k = i (each completion is one "step").
#      Variance of log s_k scales as σ²_token / L_k — much more stable
#      than per-token ratios for long completions.
#
#   2. CE-GPPO's gradient-preserving region classification:
#      in-clip  : s_k ∈ [1−ε_s, 1+ε_s]             → w = 1   (standard)
#      NA&LP    : s_k < 1−ε_s  AND  Â < 0           → w = β₁·(1−ε_s)/sg(s_k)
#      PA&LP    : s_k > 1+ε_s  AND  Â > 0           → w = β₂·(1+ε_s)/sg(s_k)
#      silenced : out-of-clip, wrong-sign advantage  → w = 0
#
# KEY: w scales ONLY the surrogate objective (s_k · Â), NOT the KL term.
#      Applying w to KL amplifies the KL penalty by 1/s_k when s_k → 0,
#      which creates a runaway feedback loop: large KL → entropy collapse →
#      s_k → 0 → w → ∞ → KL amplified further.  Keep KL unweighted.
#
# KEY: w is clamped to [0, w_max] (default 10.0) to prevent explosion when
#      s_k is very small during early training or after a bad rollout.
# ─────────────────────────────────────────────────────────────────────────────


class EarlyStopper:
    def __init__(self, patience: int = 30, min_delta: float = 0.005, warmup: int = 50):
        self.patience  = patience
        self.min_delta = min_delta
        self.warmup    = warmup
        self._best     = None
        self._no_imp   = 0
        self._calls    = 0

    def step(self, reward_ma: float):
        self._calls += 1
        if self._calls <= self.warmup:
            return False, None
        if self._best is None or reward_ma > self._best + self.min_delta:
            self._best   = reward_ma
            self._no_imp = 0
            return False, None
        self._no_imp += 1
        if self._no_imp >= self.patience:
            reason = (f"EarlyStop: reward_ma plateaued for {self.patience} log intervals "
                      f"(best={self._best:.4f}, current={reward_ma:.4f}, "
                      f"min_delta={self.min_delta})")
            return True, reason
        return False, None


def psgppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model,
                       reward_ma, kl_ma, entropy_ma, start_step=0,
                       wandb=None, early_stopper=None):
    """Returns True if early stopping was triggered, False otherwise."""
    grad_norm_val = float('nan')

    for step, batch in enumerate(loader, start=start_step + 1):

        if args.max_train_steps and step > args.max_train_steps:
            Logger(f"[STOP] Reached max_train_steps={args.max_train_steps} at step {step}.")
            if wandb and is_main_process():
                wandb.log({"train/stop_reason": "max_train_steps"})
            return True

        prompts = batch['prompt']
        prompt_inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            return_token_type_ids=False, padding_side="left", add_special_tokens=False
        ).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"]      = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

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

        rewards         = calculate_rewards(prompts, completions, args.num_generations,
                                            args.device, reward_model)
        advantages      = compute_group_advantages(rewards, args.num_generations)   # [B*G]
        completion_mask = compute_completion_mask(completion_ids, tokenizer.eos_token_id)
        seq_len         = completion_mask.float().sum(dim=1).clamp(min=1)            # [B*G]

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

        last_entropy_per_token = None
        last_kl_ref_val        = 0.0
        last_s_k               = None
        last_w                 = None
        last_in_clip           = None
        last_na_lp             = None
        last_pa_lp             = None
        last_silenced          = None
        last_token_ratio       = None
        last_policy_loss_val   = 0.0

        for k in range(args.num_policy_updates):
            with autocast_ctx:
                res      = model_unwrapped(outputs)
                aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                logits   = res.logits[:, :-1, :]
                per_token_logps = (F.log_softmax(logits, dim=-1)
                                   .gather(2, outputs[:, 1:].unsqueeze(-1))
                                   .squeeze(-1)[:, -completion_ids.size(1):])

                with torch.no_grad():
                    _lp = F.log_softmax(
                        logits[:, -completion_ids.size(1):, :].detach().float(), dim=-1)
                    entropy_per_token = -(_lp.exp() * _lp).nan_to_num(0.0).sum(dim=-1)
                    del _lp

            psgppo_out = compute_sequence_psgppo_loss(
                per_token_logps=per_token_logps,
                old_per_token_logps=old_per_token_logps,
                ref_per_token_logps=ref_per_token_logps,
                advantages=advantages,
                completion_mask=completion_mask,
                beta=args.beta,
                epsilon_s=args.epsilon_s,
                beta1=args.beta1,
                beta2=args.beta2,
                w_max=args.w_max,
                epsilon_s_mode=args.epsilon_s_mode,
                token_epsilon=args.token_epsilon,
                epsilon_s_min=args.epsilon_s_min,
                epsilon_s_max=args.epsilon_s_max,
            )
            policy_loss = psgppo_out.loss

            if args.entropy_coef > 0.0:
                _lp_g = F.log_softmax(
                    logits[:, -completion_ids.size(1):, :].float(), dim=-1)
                _ent_g = -(_lp_g.exp() * _lp_g).nan_to_num(0.0).sum(dim=-1)
                entropy_bonus = ((_ent_g * completion_mask).sum(1)
                                 / completion_mask.sum(1).clamp(1)).mean()
                del _lp_g, _ent_g
                loss = (policy_loss - args.entropy_coef * entropy_bonus + aux_loss) / args.accumulation_steps
            else:
                loss = (policy_loss + aux_loss) / args.accumulation_steps

            loss.backward()

            if (k + 1) % args.accumulation_steps == 0:
                grad_norm_val = clip_and_get_grad_norm(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            last_entropy_per_token = entropy_per_token
            last_kl_ref_val        = psgppo_out.kl_ref_val
            last_s_k               = psgppo_out.step_ratio
            last_w                 = psgppo_out.weights
            last_in_clip           = psgppo_out.in_clip
            last_na_lp             = psgppo_out.na_lp
            last_pa_lp             = psgppo_out.pa_lp
            last_silenced          = psgppo_out.silenced
            last_token_ratio       = psgppo_out.token_ratio
            last_epsilon_s         = psgppo_out.epsilon_s
            last_policy_loss_val   = policy_loss.item()

        remaining = args.num_policy_updates % args.accumulation_steps
        if remaining != 0:
            grad_norm_val = clip_and_get_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        rollout_engine.update_policy(model)

        if step % args.log_interval == 0 or step == iters:
            avg_reward = rewards.mean().item()
            avg_len    = seq_len.mean().item()

            n_steps = last_s_k.numel() + 1e-8
            in_clip_frac  = last_in_clip.float().mean().item()
            na_lp_frac    = last_na_lp.float().mean().item()
            pa_lp_frac    = last_pa_lp.float().mean().item()
            silenced_frac = last_silenced.float().mean().item()
            w_nonzero_frac = (last_w > 0).float().mean().item()
            w_mean         = last_w[last_w > 0].mean().item() if (last_w > 0).any() else 0.0

            t_stats = ratio_stats(last_token_ratio, completion_mask)
            e_stats = entropy_stats(last_entropy_per_token, completion_mask)

            reward_ma_val  = reward_ma.update(avg_reward)
            kl_ma_val      = kl_ma.update(last_kl_ref_val)
            entropy_ma_val = entropy_ma.update(e_stats["train/entropy_mean"])

            core = {
                "train/reward":           avg_reward,
                "train/policy_loss":      last_policy_loss_val,
                "train/kl_ref":           last_kl_ref_val,
                "train/avg_response_len": avg_len,
                "train/grad_norm":        grad_norm_val,
                "train/learning_rate":    optimizer.param_groups[0]['lr'],
                "train/advantages_mean":  advantages.mean().item(),
                "train/advantages_std":   advantages.std().item(),
            }
            clip_metrics = {
                **t_stats,
                # Step-level ratio stats
                "clip/step_ratio_mean":   last_s_k.mean().item(),
                "clip/step_ratio_std":    last_s_k.std().item(),
                "clip/step_ratio_p10":    last_s_k.quantile(0.10).item(),
                "clip/step_ratio_p90":    last_s_k.quantile(0.90).item(),
                "clip/epsilon_s_mean":     last_epsilon_s.mean().item(),
                "clip/epsilon_s_min":      last_epsilon_s.min().item(),
                "clip/epsilon_s_max":      last_epsilon_s.max().item(),
                # Region fractions
                "clip/step_in_clip_frac":  in_clip_frac,
                "clip/step_na_lp_frac":    na_lp_frac,
                "clip/step_pa_lp_frac":    pa_lp_frac,
                "clip/step_silenced_frac": silenced_frac,
                # Weight diagnostics — these should be ~1.0 ± small when healthy
                "clip/w_mean":             w_mean,
                "clip/w_nonzero_frac":     w_nonzero_frac,
            }
            stability = {
                "stability/reward_ma":  reward_ma_val,
                "stability/kl_ma":      kl_ma_val,
                "stability/entropy_ma": entropy_ma_val,
            }

            Logger(
                f'Ep:[{epoch+1}/{args.epochs}]({step}/{iters})  '
                f'R:{avg_reward:.3f}(ma:{reward_ma_val:.3f})  '
                f'KL:{last_kl_ref_val:.3f}(ma:{kl_ma_val:.3f})  '
                f'Loss:{last_policy_loss_val:.4f}  '
                f'Len:{avg_len:.0f}  '
                f'StepR:{last_s_k.mean().item():.5f}±{last_s_k.std().item():.5f}  '
                f'InClip:{in_clip_frac:.3f}  NA&LP:{na_lp_frac:.3f}  '
                f'PA&LP:{pa_lp_frac:.3f}  Sil:{silenced_frac:.3f}  '
                f'w_mean:{w_mean:.3f}  '
                f'Ent:{e_stats["train/entropy_mean"]:.3f}(ma:{entropy_ma_val:.3f})  '
                f'GN:{grad_norm_val:.3f}  '
                f'LR:{optimizer.param_groups[0]["lr"]:.2e}'
            )

            if wandb and is_main_process():
                log = build_log_dict(core, clip_metrics, e_stats, stability)
                log["train/outer_step"] = step
                wandb.log(log)

            if early_stopper is not None:
                stop, reason = early_stopper.step(reward_ma_val)
                if stop and is_main_process():
                    Logger(f"[EARLY STOP] {reason}")
                    if wandb:
                        wandb.log({"train/stop_reason": reason})
                    return True

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

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PS-GPPO: Per-Step Gradient-Preserving Policy Optimization")
    # ── Infrastructure (identical across ALL algorithms) ──────────────────────
    parser.add_argument("--save_dir",        type=str,   default="../out")
    parser.add_argument("--save_weight",     type=str,   default="psgppo")
    parser.add_argument("--epochs",          type=int,   default=1)
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--learning_rate",   type=float, default=3e-7)
    parser.add_argument("--device",          type=str,   default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",           type=str,   default="bfloat16")
    parser.add_argument("--num_workers",     type=int,   default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--log_interval",    type=int,   default=1)
    parser.add_argument("--save_interval",   type=int,   default=50)
    parser.add_argument("--hidden_size",     type=int,   default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe",         type=int,   default=0, choices=[0, 1])
    parser.add_argument("--max_seq_len",     type=int,   default=768)
    parser.add_argument("--max_gen_len",     type=int,   default=256)
    parser.add_argument("--data_path",       type=str,   default="../dataset/rlaif.jsonl")
    parser.add_argument("--num_generations", type=int,   default=6)
    parser.add_argument("--beta",            type=float, default=0.1)
    parser.add_argument("--from_weight",     type=str,   default="full_sft")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    parser.add_argument("--from_resume",     type=int,   default=0, choices=[0, 1])
    parser.add_argument("--use_wandb",       action="store_true")
    parser.add_argument("--wandb_project",   type=str,   default="ClipComparison")
    parser.add_argument("--wandb_mode",      type=str,   default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--use_compile",     type=int,   default=0, choices=[0, 1])
    parser.add_argument("--debug_mode",      action="store_true")
    parser.add_argument("--debug_interval",  type=int,   default=20)
    parser.add_argument("--thinking_ratio",  type=float, default=0.9)
    # ── PS-GPPO-specific parameters ───────────────────────────────────────────
    parser.add_argument("--epsilon_s",  type=float, default=0.05,
                        help="Step-level clip radius. Calibrate as ε_token/√L_avg. "
                             "For MiniMind completions (avg ~80 tokens): 0.2/√80 ≈ 0.022. "
                             "Default 0.05 is conservative; tune via clip/step_in_clip_frac.")
    parser.add_argument("--epsilon_s_mode", type=str, default="fixed",
                        choices=["fixed", "length_scaled"],
                        help="fixed uses --epsilon_s. length_scaled uses "
                             "clip(token_epsilon/sqrt(response_len), epsilon_s_min, epsilon_s_max).")
    parser.add_argument("--token_epsilon", type=float, default=0.2,
                        help="Token-level epsilon used to derive sequence epsilon in length_scaled mode.")
    parser.add_argument("--epsilon_s_min", type=float, default=0.01,
                        help="Minimum sequence epsilon in length_scaled mode.")
    parser.add_argument("--epsilon_s_max", type=float, default=0.10,
                        help="Maximum sequence epsilon in length_scaled mode.")
    parser.add_argument("--beta1",      type=float, default=0.75,
                        help="NA&LP gradient scale. <1 attenuates exploitation of unlikely "
                             "bad actions. 0 = PPO hard-zero. Best CE-GPPO result used 0.75.")
    parser.add_argument("--beta2",      type=float, default=1.0,
                        help="PA&LP gradient scale. ≥1 preserves exploration signal for "
                             "confidently-good steps. Raise to 1.1 if entropy collapses early.")
    parser.add_argument("--w_max",      type=float, default=10.0,
                        help="Hard cap on step weight w. Prevents 1/s_k explosion when s_k "
                             "is very small early in training. 10.0 allows full scaling for "
                             "s_k > beta1*(1-eps_s)/10 ≈ 0.07.")
    # ── K-epoch inner policy updates ──────────────────────────────────────────
    parser.add_argument("--num_policy_updates", type=int, default=4,
                        help="Number of gradient steps per rollout batch. "
                             "K=1 gives s_k≡1. K≥2 enables step-level regions to activate.")
    # ── Experiment length controls ────────────────────────────────────────────
    parser.add_argument("--max_train_steps", type=int, default=0,
                        help="Hard cap on outer steps (0 = no cap).")
    # ── Early stopping ─────────────────────────────────────────────────────────
    parser.add_argument("--early_stop",           action="store_true")
    parser.add_argument("--early_stop_patience",  type=int,   default=30)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.005)
    parser.add_argument("--early_stop_warmup",    type=int,   default=50)
    # ── Entropy ───────────────────────────────────────────────────────────────
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="Entropy bonus coefficient (0 = disabled). "
                             "Try 0.01 if clip/step_in_clip_frac stays near 0.")

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
        wandb = init_wandb_safely(
            project=args.wandb_project,
            name=(f"ps-gppo-eps{args.epsilon_s}"
                  f"-b1{args.beta1}-b2{args.beta2}"
                  f"-K{args.num_policy_updates}-LR{args.learning_rate}"),
            config=vars(args),
            wandb_id=ckp_data.get('wandb_id') if ckp_data else None,
            mode=args.wandb_mode,
        )

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
    total_steps   = math.ceil(iters / args.accumulation_steps) * args.epochs * args.num_policy_updates
    scheduler     = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1),
                                      eta_min=args.learning_rate / 10)

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

    reward_ma  = MovingAverage(alpha=0.05)
    kl_ma      = MovingAverage(alpha=0.05)
    entropy_ma = MovingAverage(alpha=0.10)

    early_stopper = None
    if args.early_stop:
        early_stopper = EarlyStopper(
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            warmup=args.early_stop_warmup,
        )

    if is_main_process():
        Logger(
            f"[CONFIG] PS-GPPO | eps_s={args.epsilon_s} | "
            f"eps_mode={args.epsilon_s_mode} token_eps={args.token_epsilon} | "
            f"beta1={args.beta1} beta2={args.beta2} w_max={args.w_max} | "
            f"K={args.num_policy_updates} | max_steps={args.max_train_steps or 'none'} | "
            f"max_gen_len={args.max_gen_len} | entropy_coef={args.entropy_coef} | "
            f"early_stop={args.early_stop}"
        )

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
        stopped = psgppo_train_epoch(epoch, loader, n_iters, rollout_engine, ref_model, reward_model,
                                     reward_ma, kl_ma, entropy_ma, skip, wandb, early_stopper)
        if stopped:
            break

    if dist.is_initialized(): dist.destroy_process_group()
