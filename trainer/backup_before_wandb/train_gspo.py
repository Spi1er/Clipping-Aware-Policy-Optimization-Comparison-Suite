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
# GSPO: Group Sequence Policy Optimization
# arXiv: 2507.18071
#
# Core motivation: GRPO's token-level importance weights are theoretically
# ill-posed because importance sampling requires averaging over many samples,
# but each token weight is based on a single draw from π_old(·|x, y<t).
# This injects high-variance noise that accumulates over long sequences and
# is amplified by the token-level clipping, causing catastrophic collapse in
# large (especially MoE) models.
#
# Key innovation: sequence-level importance ratio with length normalisation.
#
#   s_i(θ) = [ π_θ(y_i|x) / π_old(y_i|x) ]^(1/|y_i|)
#           = exp( 1/|y_i| · Σ_t  log[π_θ(y_i,t|x,y_i,<t) / π_old(y_i,t|x,y_i,<t)] )
#
# The 1/|y_i| exponent normalises by length so that the ratio is on a
# consistent numerical scale regardless of response length, dramatically
# reducing variance for long outputs.
#
# Objective (Eq. 5 in the paper):
#
#   J_GSPO(θ) = E[ 1/G Σ_i  min( s_i(θ)·Â_i,  clip(s_i(θ), 1−ε, 1+ε)·Â_i ) ]
#
# where Â_i = (R_i − mean(R)) / std(R) is the group-normalised advantage
# (identical to GRPO's advantage, but applied at sequence level).
#
# Gradient flow:
#   The loss for sequence i is a scalar f(s_i) · Â_i.
#   s_i = exp(mean_t log_r_i,t), so ∂s_i/∂log_π_θ(y_i,t) = s_i / |y_i|.
#   Therefore each token in y_i receives an equal gradient proportional to
#   s_i · Â_i / |y_i|, preserving the alignment between sequence reward and
#   token-level parameter update.
#
# Implementation notes:
#   1. Compute per-token log-ratios log_r_i,t = log_π_θ − log_π_old.
#   2. Mask padding; average masked log-ratios → seq_log_ratio.
#   3. s_i = exp(seq_log_ratio).
#   4. Apply standard PPO min-clip at sequence level → scalar loss per seq.
#   5. Average over sequences in the batch.
#   6. Add KL penalty at token level (optional, controlled by --beta).
#
# GSPO resolves MoE instability because it never amplifies per-token
# variance via the token-level clip: one clipping decision per sequence
# instead of one per token.
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


def gspo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model,
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

        # ---- Group-normalised advantages (same as GRPO) ----
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, G]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r  = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)         # [B*G]

        # ---- Completion mask ----
        is_eos = completion_ids == tokenizer.eos_token_id         # [B*G, L]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device)
                           .expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).float()  # [B*G, L]

        # ---- KL divergence (token-level, for regularisation) ----
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1            # [B*G, L]

        # ---- GSPO: sequence-level importance ratio ----
        # log r_i,t = log π_θ(y_i,t) - log π_old(y_i,t)
        log_ratio_per_token = per_token_logps - old_per_token_logps  # [B*G, L]

        # Length-normalised log ratio: 1/|y_i| · Σ_t mask_t · log_r_i,t
        seq_len = completion_mask.sum(dim=1).clamp(min=1)            # [B*G]
        seq_log_ratio = (log_ratio_per_token * completion_mask).sum(dim=1) / seq_len  # [B*G]

        # Sequence-level importance ratio s_i = exp(seq_log_ratio)
        seq_ratio = seq_log_ratio.exp()                               # [B*G]

        # PPO-style min-clip at sequence level
        adv = advantages                                              # [B*G]
        clipped_seq_ratio = torch.clamp(seq_ratio, 1.0 - args.epsilon, 1.0 + args.epsilon)
        seq_obj = torch.min(seq_ratio * adv, clipped_seq_ratio * adv)  # [B*G]

        # Token-level KL penalty averaged per sequence
        seq_kl = (per_token_kl * completion_mask).sum(dim=1) / seq_len  # [B*G]

        # Final per-sequence loss
        seq_loss = -(seq_obj - args.beta * seq_kl)                   # [B*G]
        policy_loss = seq_loss.mean()
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
            avg_len_val         = seq_len.mean().item()
            kl_ref_val          = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() \
                                  / completion_mask.sum().item()
            advantages_mean_val = advantages.mean().item()
            advantages_std_val  = advantages.std().item()
            avg_seq_ratio       = seq_ratio.mean().item()
            clipped_frac        = (seq_ratio != clipped_seq_ratio).float().mean().item()
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {loss.item()*args.accumulation_steps:.4f}, '
                   f'Seq Ratio: {avg_seq_ratio:.4f}, Clipped: {clipped_frac:.3f}, '
                   f'Avg Len: {avg_len_val:.2f}, LR: {optimizer.param_groups[0]["lr"]:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward": avg_reward_val, "kl_ref": kl_ref_val,
                           "policy_loss": loss.item() * args.accumulation_steps,
                           "avg_seq_ratio": avg_seq_ratio,
                           "seq_clipped_fraction": clipped_frac,
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
    parser = argparse.ArgumentParser(description="GSPO: Group Sequence Policy Optimization")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='gspo', type=str)
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
    # GSPO uses the same epsilon as PPO clip, but applied at sequence level
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="Sequence-level clip range for s_i = (π_θ/π_old)^(1/|y|). "
                             "Must be much smaller than token-level ε (paper uses 3e-4 to 4e-4 "
                             "for 30B model; 0.005 is a reasonable starting point for smaller models). "
                             "Using 0.2 like GRPO would make clipping effectively inactive.")
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GSPO")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=20)
    parser.add_argument("--thinking_ratio", type=float, default=0.9)
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # GSPO epsilon sanity check.
    # The sequence-level ratio s_i = (π_θ/π_old)^(1/|y|) lives near 1 with much tighter
    # spread than token-level ratios. Using ε=0.2 (GRPO default) would make clipping
    # effectively inactive. The original paper (30B model) uses ε ≈ 3e-4 to 4e-4.
    # For smaller models / shorter sequences 0.001–0.01 is a reasonable range.
    if args.epsilon > 0.05 and is_main_process():
        Logger(f"[GSPO WARNING] --epsilon={args.epsilon} looks too large for a sequence-level ratio. "
               f"GSPO clips s_i=(π_θ/π_old)^(1/|y|), not per-token ratios. "
               f"Recommended range: 0.001–0.01. The paper uses 3e-4 to 4e-4 for a 30B model.")

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
                   name=f"GSPO-Epoch{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}",
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
            gspo_train_epoch(epoch, loader, len(loader)+skip, rollout_engine, ref_model, reward_model,
                             start_step, wandb)
        else:
            gspo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model,
                             0, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
