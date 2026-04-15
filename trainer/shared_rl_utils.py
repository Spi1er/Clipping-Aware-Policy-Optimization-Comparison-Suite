"""
shared_rl_utils.py — Shared RL training infrastructure for the clipping-comparison suite.

Algorithms covered: GRPO · GSPO · DAPO · DCPO · CFPO · SAPO · CE-GPPO

Everything in this file is IDENTICAL across all algorithms.  The algorithm-specific
surrogate-objective code stays in each train_*.py.  Keeping shared logic here makes
it easy to audit that comparisons are fair:

    "Any difference you see in experiment results comes from the objective function,
     not from inconsistent infrastructure."

Canonical W&B key schema (all algorithms log NaN for inapplicable metrics):
    train/  — core training curves
    clip/   — ratio / clipping behaviour
    stability/ — moving averages for smoothed curves
"""

import re
import torch
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def rep_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    """N-gram repetition penalty — shared across all algorithms."""
    toks  = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, num_generations: int, device, reward_model) -> torch.Tensor:
    """
    Compute rewards for B*G responses under the shared reward function.

    Reward components (same for all algorithms):
      +0.5  if 20 ≤ len(response) ≤ 800, else -0.5
      +1.0  for well-formed thinking (20–300 chars), else -0.5  (if </think> present)
      +0.25 if exactly one </think> tag, else -0.25
      − rep_penalty(answer)
      + reward_model.get_score(messages, answer)   (clamped to [-3, 3] inside scorer)

    Args:
        prompts:         list[str]  length B
        responses:       list[str]  length B*G
        num_generations: G
        device:          torch device
        reward_model:    LMForRewardModel instance (shared object)

    Returns:
        rewards: Tensor [B*G]
    """
    rewards = torch.zeros(len(responses), device=device)
    with torch.no_grad():
        scores = []
        for i in range(len(prompts)):
            for j in range(num_generations):
                idx      = i * num_generations + j
                response = responses[idx]
                prompt   = prompts[i]

                pattern  = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches  = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer   = response

                rewards[idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[idx] -= rep_penalty(answer)
                scores.append(reward_model.get_score(messages, answer))

        rewards += torch.tensor(scores, device=device)
    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETION MASK
# ═══════════════════════════════════════════════════════════════════════════════

def compute_completion_mask(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Binary mask [B*G, L]: 1 for every token up to and including the first EOS.

    This exact logic is used identically by all algorithms to ensure that
    loss aggregation, KL, and ratio statistics are computed over the same
    tokens.
    """
    device  = completion_ids.device
    is_eos  = completion_ids == eos_token_id                               # [B*G, L]
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    mask = (torch.arange(is_eos.size(1), device=device)
            .expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()    # [B*G, L]
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANTAGE NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_group_advantages(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """
    Standard group-relative advantage normalisation (GRPO / DAPO / CFPO / SAPO / CE-GPPO).

    Each prompt's G completions form a group.  Advantages are z-scored within
    each group so the mean is 0 and std ≈ 1.

    Note: DCPO replaces this with Smooth Advantage Standardisation (SAS) — that
    logic stays in train_dcpo.py because it maintains per-prompt Welford state.
    """
    grouped = rewards.view(-1, num_generations)                            # [B, G]
    mean_r  = grouped.mean(dim=1).repeat_interleave(num_generations)       # [B*G]
    std_r   = grouped.std(dim=1).repeat_interleave(num_generations)        # [B*G]
    return (rewards - mean_r) / (std_r + 1e-4)                            # [B*G]


# ═══════════════════════════════════════════════════════════════════════════════
# KL METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kl_terms(ref_per_token_logps: torch.Tensor,
                     per_token_logps: torch.Tensor,
                     completion_mask: torch.Tensor):
    """
    Returns:
        per_token_kl  [B*G, L]  — regularisation term used in the loss: exp(d)−d−1
        kl_ref_val    float     — log-space KL proxy for logging
                                  = mean_t[ log(π_ref/π_θ) ]  ≈ E[log(π_ref/π_θ)]

    Both computed identically in all algorithms.  The 'kl_ref_val' logged to
    W&B is the same quantity across methods, so KL curves are directly comparable.
    """
    kl_div       = ref_per_token_logps - per_token_logps              # log(π_ref/π_θ)
    per_token_kl = torch.exp(kl_div) - kl_div - 1                     # for loss (non-negative)
    kl_ref_val   = (kl_div * completion_mask).sum().item() / (completion_mask.sum().item() + 1e-8)
    return per_token_kl, kl_ref_val


# ═══════════════════════════════════════════════════════════════════════════════
# RATIO STATISTICS  (for W&B clipping-behaviour group)
# ═══════════════════════════════════════════════════════════════════════════════

def ratio_stats(ratio: torch.Tensor, completion_mask: torch.Tensor) -> Dict[str, float]:
    """
    Token-level importance ratio statistics — prefixed clip/.
    Used by GRPO, DAPO, DCPO, CFPO, SAPO, CE-GPPO.
    GSPO uses seq_ratio_stats() for its sequence-level ratio.
    """
    r = ratio[completion_mask.bool()]
    return {
        "clip/ratio_mean": r.mean().item(),
        "clip/ratio_std":  r.std().item(),
        "clip/ratio_min":  r.min().item(),
        "clip/ratio_max":  r.max().item(),
        "clip/ratio_p10":  r.quantile(0.10).item(),
        "clip/ratio_p90":  r.quantile(0.90).item(),
    }


def seq_ratio_stats(seq_ratio: torch.Tensor) -> Dict[str, float]:
    """
    Sequence-level importance ratio statistics for GSPO.
    s_i = exp(1/|y_i| · Σ_t log(π_θ/π_old)) — lives much closer to 1 than
    token-level ratios, so uses a dedicated key namespace to avoid confusion.
    """
    return {
        "clip/seq_ratio_mean": seq_ratio.mean().item(),
        "clip/seq_ratio_std":  seq_ratio.std().item(),
        "clip/seq_ratio_min":  seq_ratio.min().item(),
        "clip/seq_ratio_max":  seq_ratio.max().item(),
        "clip/seq_ratio_p10":  seq_ratio.quantile(0.10).item(),
        "clip/seq_ratio_p90":  seq_ratio.quantile(0.90).item(),
    }


def clipped_fractions(ratio: torch.Tensor,
                      advantages_flat: torch.Tensor,
                      completion_mask: torch.Tensor,
                      eps_low: float, eps_high: float) -> Dict[str, float]:
    """
    Fraction of valid (masked) tokens that fall outside the clip bounds,
    further split by advantage sign.

    Meaningful for hard-clip methods: GRPO, DAPO, DCPO, CE-GPPO.
    For CFPO/SAPO the clipping analogue is logged separately in those scripts.

    eps_low:  how far below 1 the lower bound is  (ratio < 1 − eps_low  → below)
    eps_high: how far above 1 the upper bound is  (ratio > 1 + eps_high → above)
    """
    m   = completion_mask.bool()
    adv = advantages_flat.unsqueeze(1).expand_as(ratio)               # [B*G, L]

    above = ratio > (1.0 + eps_high)
    below = ratio < (1.0 - eps_low)
    any_c = above | below

    total = m.float().sum().item() + 1e-8
    return {
        # Overall clipping rate
        "clip/clipped_fraction":     (any_c & m).float().sum().item() / total,
        "clip/above_clip_fraction":  (above & m).float().sum().item() / total,
        "clip/below_clip_fraction":  (below & m).float().sum().item() / total,
        # Split by advantage sign  (NA&LP / PA&LP in CE-GPPO language)
        "clip/pos_adv_clipped_frac": ((above & (adv > 0)) & m).float().sum().item() / total,
        "clip/neg_adv_clipped_frac": ((below & (adv <= 0)) & m).float().sum().item() / total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MOVING AVERAGE  (for stability curves)
# ═══════════════════════════════════════════════════════════════════════════════

class MovingAverage:
    """Exponential moving average for reward and KL stability curves."""

    def __init__(self, alpha: float = 0.05):
        """alpha: smoothing factor (smaller = slower adaptation)."""
        self.alpha = alpha
        self._val: Optional[float] = None

    def update(self, x: float) -> float:
        if self._val is None:
            self._val = x
        else:
            self._val = (1.0 - self.alpha) * self._val + self.alpha * x
        return self._val

    @property
    def value(self) -> Optional[float]:
        return self._val


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENT NORM HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def clip_and_get_grad_norm(model_parameters, grad_clip: float) -> float:
    """
    Clip gradients and return the pre-clip global gradient norm.
    If grad_clip <= 0, computes the norm without clipping.
    Logging the pre-clip norm makes it easy to see when clipping is active.
    """
    if grad_clip > 0:
        return torch.nn.utils.clip_grad_norm_(model_parameters, grad_clip).item()
    else:
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model_parameters
            if p.grad is not None
        )
        return total ** 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# W&B LOGGING SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

# Full set of canonical keys — ALL algorithms should emit ALL keys.
# Inapplicable keys are filled with float('nan') so W&B curves stay aligned.
_ALL_KEYS = [
    # ── core training curves ──────────────────────────────────────────────────
    "train/reward", "train/policy_loss", "train/kl_ref",
    "train/avg_response_len", "train/grad_norm", "train/learning_rate",
    "train/advantages_mean", "train/advantages_std",
    # ── universal token-level ratio stats ─────────────────────────────────────
    "clip/ratio_mean", "clip/ratio_std",
    "clip/ratio_min", "clip/ratio_max",
    "clip/ratio_p10", "clip/ratio_p90",
    # ── hard-clip fractions (GRPO / DAPO / DCPO / CE-GPPO) ───────────────────
    "clip/clipped_fraction",
    "clip/above_clip_fraction", "clip/below_clip_fraction",
    "clip/pos_adv_clipped_frac", "clip/neg_adv_clipped_frac",
    # ── GSPO: sequence-level ratio ────────────────────────────────────────────
    "clip/seq_ratio_mean", "clip/seq_ratio_std",
    "clip/seq_ratio_min", "clip/seq_ratio_max",
    "clip/seq_ratio_p10", "clip/seq_ratio_p90",
    "clip/seq_clipped_fraction",
    # ── DCPO: dynamic adaptive clip bounds ────────────────────────────────────
    "clip/lower_bound_mean", "clip/upper_bound_mean",
    "clip/dyn_clipped_fraction",
    # ── CFPO: clipping-free penalty ───────────────────────────────────────────
    "clip/quadratic_penalty_mean", "clip/off_policy_fraction",
    # ── SAPO: soft gate ───────────────────────────────────────────────────────
    "clip/avg_gate", "clip/gate_saturation_frac",
    # ── CE-GPPO: gradient-preserving regions ─────────────────────────────────
    "clip/nalp_fraction", "clip/palp_fraction", "clip/preserved_grad_frac",
    # ── stability / smoothed curves ───────────────────────────────────────────
    "stability/reward_ma", "stability/kl_ma",
]


def build_log_dict(*metric_dicts) -> dict:
    """
    Merge metric dicts and fill every missing canonical key with NaN.

    Usage:
        wandb.log(build_log_dict(core_metrics, clip_metrics, stability_metrics))

    The NaN fill keeps all W&B series aligned — you can overlay any two
    algorithms on the same plot without missing-series gaps.
    """
    log = {k: float('nan') for k in _ALL_KEYS}
    for d in metric_dicts:
        if d:
            log.update(d)
    return log
