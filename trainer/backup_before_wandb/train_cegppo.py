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
# CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization
# arXiv: 2509.20712
#
# Key innovation: instead of zeroing out gradients from tokens outside the PPO
# clip interval, CE-GPPO reintroduces them at a bounded magnitude.
#
# Token categories (based on ratio r and advantage A):
#   PA&HP  (r in clip, A>0)  — in-clip, standard update
#   NA&HP  (r in clip, A<0)  — in-clip, standard update
#   PA&LP  (r > 1+ε, A>0)   — above clip: gradient normally zeroed by PPO
#   NA&LP  (r < 1−ε, A<0)   — below clip: gradient normally zeroed by PPO
#
# Per-token loss function (from Eq. 8 in the paper):
#
#   l_t = beta1 * (1−ε) / sg(r_t) * r_t * A      if r_t < 1−ε  AND  A < 0   (NA&LP)
#   l_t = beta2 * (1+ε) / sg(r_t) * r_t * A      if r_t > 1+ε  AND  A > 0   (PA&LP)
#   l_t = r_t * A                                  otherwise  (in-clip OR PPO's passthrough)
#
# Using sg (stop-gradient) on the denominator means the gradient w.r.t. log π_θ is:
#   ∇ log π_θ * A * beta1*(1−ε)   for NA&LP
#   ∇ log π_θ * A * beta2*(1+ε)   for PA&LP
# — bounded values instead of zero, so entropy is explicitly regulated.
#
# Setting beta1=beta2=0 recovers standard PPO (but without the min() clipping).
# Typical values: beta1=0.75, beta2=1.0  (more weight on PA&LP = preserve exploration).
#
# Token-level loss: normalise by total tokens across the whole group.
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


def cegppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model,
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

        # ---- CE-GPPO loss (Eq. 8 in the paper) ----
        # Base: unclipped ratio * advantage (for all "otherwise" tokens)
        per_token_obj = ratio * adv

        # Identify NA&LP tokens: below lower clip AND negative advantage
        mask_nalp = (ratio < (1.0 - args.epsilon)) & (adv < 0.0)
        # Identify PA&LP tokens: above upper clip AND positive advantage
        mask_palp = (ratio > (1.0 + args.epsilon)) & (adv > 0.0)

        # For NA&LP: l = beta1 * (1-ε) / sg(ratio) * ratio * A
        # stop_gradient on ratio in denominator → gradient = beta1*(1-ε)*A * ∇log_π
        nalp_obj = args.beta1_ce * (1.0 - args.epsilon) / ratio.detach() * ratio * adv

        # For PA&LP: l = beta2 * (1+ε) / sg(ratio) * ratio * A
        # gradient = beta2*(1+ε)*A * ∇log_π
        palp_obj = args.beta2_ce * (1.0 + args.epsilon) / ratio.detach() * ratio * adv

        per_token_obj = torch.where(mask_nalp, nalp_obj,
                        torch.where(mask_palp, palp_obj, per_token_obj))

        per_token_loss = -(per_token_obj - args.beta * per_token_kl)

        # Loss aggregation: per-sequence mean → batch mean (same as train_grpo.py baseline)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if is_main_process() and step % args.save_interval == 0:
                rollout_engine.update_policy(model)

        if step % args.log_interval == 0 or step == iters:
            avg_reward_val = rewards.mean().item()
            avg_len_val    = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val     = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() \
                             / completion_mask.sum().item()
            advantages_mean_val = advantages.mean().item()
            advantages_std_val  = advantages.std().item()
            total_tokens = completion_mask.sum()
            nalp_frac = (mask_nalp * completion_mask.bool()).float().sum().item() / (total_tokens.item() + 1e-8)
            palp_frac = (mask_palp * completion_mask.bool()).float().sum().item() / (total_tokens.item() + 1e-8)
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {loss.item()*args.accumulation_steps:.4f}, '
                   f'NA&LP: {nalp_frac:.3f}, PA&LP: {palp_frac:.3f}, '
                   f'Avg Len: {avg_len_val:.2f}, LR: {optimizer.param_groups[0]["lr"]:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward": avg_reward_val, "kl_ref": kl_ref_val,
                           "policy_loss": loss.item() * args.accumulation_steps,
                           "nalp_fraction": nalp_frac, "palp_fraction": palp_frac,
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
    parser = argparse.ArgumentParser(
        description="CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='cegppo', type=str)
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
    parser.add_argument("--epsilon", type=float, default=0.2, help="Symmetric clip range")
    # CE-GPPO-specific: gradient scaling for out-of-clip tokens
    parser.add_argument("--beta1_ce", type=float, default=0.75,
                        help="Scale for NA&LP gradients (<1 slows entropy, =0 is PPO, >1 boosts exploitation)")
    parser.add_argument("--beta2_ce", type=float, default=1.0,
                        help="Scale for PA&LP gradients (>0 preserves exploration, =0 is PPO)")
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-CEGPPO")
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
                   name=f"CEGPPO-Epoch{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}",
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
            cegppo_train_epoch(epoch, loader, len(loader)+skip, rollout_engine, ref_model, reward_model,
                               start_step, wandb)
        else:
            cegppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model,
                               0, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
