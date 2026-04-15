import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CFPO: Clipping-Free Policy Optimization
# arXiv: 2601.22801
#
# Key innovation: replaces hard PPO/GRPO clipping with a convex quadratic
# penalty derived from Total Variation (TV) divergence constraints.
#
# Per-token objective (Eq. 7 in the paper):
#
#   L_CFPO(r, A) = r * A  −  |A| / (2ε) * (r − 1)²
#
# where r = π_θ / π_old is the per-token probability ratio.
#
# Properties:
#   • Everywhere differentiable: gradient is always −|A|/ε * (r−1), which
#     provides a continuous restoring force pulling r back toward 1.
#   • No zero-gradient regions: clipping creates flat regions beyond
#     [1−ε, 1+ε]; CFPO has a gradient everywhere.
#   • Convex in r: the quadratic penalty is minimised at r=1 and grows
#     quadratically with deviation, so large off-policy updates are penalised
#     proportionally to |A| (larger advantages permit larger updates).
#   • One-line code change from GRPO:
#       GRPO: min(r*A, clip(r, 1±ε)*A)
#       CFPO: r*A − |A|/(2ε) * (r−1)²
#
# Full CFPO objective (Eq. 8):
#   J = E[ 1/G Σ_i 1/|o_i| Σ_t (r_t*A − |A|/(2ε)*(r_t−1)² − β*KL) ]
#
# Implementation note:
#   The gradient of CFPO w.r.t. log π_θ is:
#     (A − |A|/ε * (r−1) * r) * ∇ log π_θ
#   At r=1: (A − 0) * ∇ log π_θ = A * ∇ log π_θ  (same as REINFORCE).
#   For r far from 1: the quadratic term pulls back toward on-policy.
# ---------------------------------------------------------------------------


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)
    with torch.no_grad():
        scores = []
        for i in range(len(prompts)):
            for j in range(args.num_generations):
                idx = i * args.num_generations + j
                response = responses[idx]
                prompt = prompts[i]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                rewards[idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[idx] -= rep_penalty(answer)
                scores.append(reward_model.get_score(messages, answer))
        rewards += torch.tensor(scores, device=args.device)
    return rewards


def cfpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model,
                     start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                                  return_token_type_ids=False, padding_side="left",
                                  add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs          = rollout_result.output_ids
        completion_ids   = rollout_result.completion_ids
        completions      = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]
            per_token_logps = F.log_softmax(logits, dim=-1).gather(
                2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -completion_ids.size(1):]

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))

        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r  = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device)
                           .expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*G, L]
        adv   = advantages.unsqueeze(1)                           # [B*G, 1]

        # ---- CFPO loss (Eq. 7 in the paper) ----
        # L = r*A − |A|/(2ε) * (r−1)²  − β*KL
        # No clipping of any kind. The quadratic term acts as the trust region.
        quadratic_penalty = adv.abs() / (2.0 * args.epsilon) * (ratio - 1.0) ** 2
        per_token_obj  = ratio * adv - quadratic_penalty
        per_token_loss = -(per_token_obj - args.beta * per_token_kl)

        # Standard per-sequence mean, then batch mean (same as GRPO aggregation)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) \
                       / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if is_main_process() and step % args.save_interval == 0:
                rollout_engine.update_policy(model)

        if step % args.log_interval == 0 or step == iters:
            avg_reward_val      = rewards.mean().item()
            avg_len_val         = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val          = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() \
                                  / completion_mask.sum().item()
            advantages_mean_val = advantages.mean().item()
            advantages_std_val  = advantages.std().item()
            # Fraction of tokens where ratio deviates > ε (equivalent to what GRPO would clip)
            completion_mask_bool = completion_mask.bool()
            off_policy_frac = ((ratio - 1.0).abs() > args.epsilon)[completion_mask_bool].float().mean().item()
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {loss.item()*args.accumulation_steps:.4f}, '
                   f'Off-Policy Frac: {off_policy_frac:.3f}, '
                   f'Avg Len: {avg_len_val:.2f}, LR: {optimizer.param_groups[0]["lr"]:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward": avg_reward_val, "kl_ref": kl_ref_val,
                           "policy_loss": loss.item() * args.accumulation_steps,
                           "off_policy_fraction": off_policy_frac,
                           "avg_response_len": avg_len_val,
                           "learning_rate": optimizer.param_groups[0]['lr']})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        if is_main_process() and step % args.save_interval == 0:
            rollout_engine.update_policy(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFPO: Clipping-Free Policy Optimization")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='cfpo', type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl")
    parser.add_argument("--num_generations", type=int, default=6)
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    # CFPO: epsilon controls the trust region width for the quadratic penalty
    # Equivalent to the ε in PPO clipping – larger = more permissive updates
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Trust region width ε: penalty = |A|/(2ε)*(r-1)^2")
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-CFPO")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=20)
    parser.add_argument("--thinking_ratio", type=float, default=0.9)
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        wandb.init(project=args.wandb_project,
                   name=f"CFPO-Epoch{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}",
                   id=wandb_id, resume='must' if wandb_id else None)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    rollout_engine = create_rollout_engine(
        engine_type="torch", policy_model=model, tokenizer=tokenizer,
        device=args.device, autocast_ctx=autocast_ctx,
    )

    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len,
                            thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    if is_main_process(): rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: resuming from step {start_step+1}')
            cfpo_train_epoch(epoch, loader, len(loader)+skip, rollout_engine, ref_model, reward_model,
                             start_step, wandb)
        else:
            cfpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model,
                             0, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
