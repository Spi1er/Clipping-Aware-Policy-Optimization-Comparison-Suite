"""Sequence-level PS-GPPO objective for MiniMind RLAIF training.

The standalone PS-GPPO package is designed for multi-turn agentic RL, where each
trajectory has turn_ids and one advantage per agent step.  The MiniMind RLAIF
dataset is single-turn: one prompt, one sampled completion, one reward.  This
module implements the honest single-turn analogue:

    one completion == one step

It is still useful as an experimental bridge because it tests the two mechanisms
that motivated PS-GPPO:

1. geometric-mean sequence ratio instead of raw token-level ratios;
2. CE-GPPO-style gradient preservation outside the clip window.

For true agentic RL claims, use the separate PS-GPPO package with multi-turn
rollouts, turn_ids, and step-level/process advantages.
"""

from dataclasses import dataclass
from typing import Literal

import torch


EpsilonMode = Literal["fixed", "length_scaled"]


@dataclass
class SequencePSGPPOLossOutput:
    loss: torch.Tensor
    kl_ref_val: float
    per_token_kl: torch.Tensor
    token_ratio: torch.Tensor
    step_ratio: torch.Tensor
    epsilon_s: torch.Tensor
    weights: torch.Tensor
    in_clip: torch.Tensor
    na_lp: torch.Tensor
    pa_lp: torch.Tensor
    silenced: torch.Tensor


def resolve_epsilon_s(
    seq_len: torch.Tensor,
    epsilon_s: float,
    epsilon_s_mode: EpsilonMode = "fixed",
    token_epsilon: float = 0.2,
    epsilon_s_min: float = 0.01,
    epsilon_s_max: float = 0.10,
) -> torch.Tensor:
    """Return one step-level clip radius per sampled completion.

    fixed:
        Uses the scalar epsilon_s for every completion.

    length_scaled:
        Uses token_epsilon / sqrt(valid_completion_tokens), clamped to a
        reasonable interval.  This follows the geometric-mean ratio intuition:
        the standard deviation of log step-ratio scales roughly as 1/sqrt(L).
    """
    if epsilon_s_mode == "fixed":
        return torch.full_like(seq_len, float(epsilon_s), dtype=torch.float32)
    if epsilon_s_mode == "length_scaled":
        eps = float(token_epsilon) / torch.sqrt(seq_len.float().clamp(min=1.0))
        return eps.clamp(min=float(epsilon_s_min), max=float(epsilon_s_max))
    raise ValueError(f"Unknown epsilon_s_mode: {epsilon_s_mode}")


def compute_sequence_psgppo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    beta: float,
    epsilon_s: float = 0.05,
    beta1: float = 0.75,
    beta2: float = 1.0,
    w_max: float = 10.0,
    epsilon_s_mode: EpsilonMode = "fixed",
    token_epsilon: float = 0.2,
    epsilon_s_min: float = 0.01,
    epsilon_s_max: float = 0.10,
) -> SequencePSGPPOLossOutput:
    """Compute the sequence-level PS-GPPO loss for one RLAIF rollout batch.

    Shapes:
        per_token_logps:     [B*G, L]
        old_per_token_logps: [B*G, L]
        ref_per_token_logps: [B*G, L]
        advantages:          [B*G]
        completion_mask:     [B*G, L]

    Returns a dataclass with the scalar loss and tensors needed for diagnostics.
    """
    if per_token_logps.shape != old_per_token_logps.shape:
        raise ValueError("per_token_logps and old_per_token_logps must have the same shape")
    if per_token_logps.shape != ref_per_token_logps.shape:
        raise ValueError("per_token_logps and ref_per_token_logps must have the same shape")
    if completion_mask.shape != per_token_logps.shape:
        raise ValueError("completion_mask must match logprob shape")

    mask = completion_mask.float()
    adv = advantages.reshape(-1).to(per_token_logps.device, dtype=per_token_logps.dtype)
    if adv.numel() != per_token_logps.size(0):
        raise ValueError("advantages must have one value per sampled completion")

    seq_len = mask.sum(dim=1).clamp(min=1.0)
    eps = resolve_epsilon_s(
        seq_len=seq_len,
        epsilon_s=epsilon_s,
        epsilon_s_mode=epsilon_s_mode,
        token_epsilon=token_epsilon,
        epsilon_s_min=epsilon_s_min,
        epsilon_s_max=epsilon_s_max,
    ).to(per_token_logps.device, dtype=per_token_logps.dtype)

    log_ratio = (per_token_logps - old_per_token_logps) * mask
    log_s = log_ratio.sum(dim=1) / seq_len
    step_ratio = torch.exp(log_s.clamp(min=-20.0, max=20.0))
    token_ratio = torch.exp((per_token_logps - old_per_token_logps).clamp(min=-20.0, max=20.0))

    lower = 1.0 - eps
    upper = 1.0 + eps
    in_clip = (step_ratio >= lower) & (step_ratio <= upper)
    na_lp = (step_ratio < lower) & (adv < 0.0)
    pa_lp = (step_ratio > upper) & (adv > 0.0)
    silenced = ~(in_clip | na_lp | pa_lp)

    weights = torch.ones_like(step_ratio)
    if na_lp.any():
        weights[na_lp] = (
            float(beta1) * lower[na_lp] / step_ratio[na_lp].detach().clamp(min=1e-6)
        ).clamp(max=float(w_max))
    if pa_lp.any():
        weights[pa_lp] = (
            float(beta2) * upper[pa_lp] / step_ratio[pa_lp].detach().clamp(min=1e-6)
        ).clamp(max=float(w_max))
    weights[silenced] = 0.0

    kl_div = ref_per_token_logps - per_token_logps
    per_token_kl = (torch.exp(kl_div.clamp(min=-20.0, max=20.0)) - kl_div - 1.0) * mask
    kl_ref_val = ((kl_div * mask).sum() / mask.sum().clamp(min=1.0)).detach().float().item()

    surrogate = (step_ratio * adv * weights).unsqueeze(1)
    per_token_loss = -(surrogate - float(beta) * per_token_kl)
    loss = ((per_token_loss * mask).sum(dim=1) / seq_len).mean()

    return SequencePSGPPOLossOutput(
        loss=loss,
        kl_ref_val=kl_ref_val,
        per_token_kl=per_token_kl,
        token_ratio=token_ratio,
        step_ratio=step_ratio,
        epsilon_s=eps.detach(),
        weights=weights.detach(),
        in_clip=in_clip.detach(),
        na_lp=na_lp.detach(),
        pa_lp=pa_lp.detach(),
        silenced=silenced.detach(),
    )
