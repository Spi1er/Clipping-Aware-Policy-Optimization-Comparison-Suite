"""CPU smoke test for the MiniMind sequence-level PS-GPPO objective.

This test does not load MiniMind weights or the reward model.  It only verifies
the objective math: finite loss, clean backward pass, region metrics, and the
optional length-scaled epsilon path.

Run from the MiniMind root or trainer directory:

    python trainer/smoke_test_psgppo_sequence.py
    python smoke_test_psgppo_sequence.py
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from trainer.algorithms.psgppo_sequence import compute_sequence_psgppo_loss


def main():
    torch.manual_seed(7)
    batch, length = 6, 17

    old_logps = torch.randn(batch, length) * 0.05 - 1.0
    current_logps = (old_logps + torch.randn(batch, length) * 0.04).detach().requires_grad_(True)
    ref_logps = old_logps.detach() + torch.randn(batch, length) * 0.02

    mask = torch.ones(batch, length)
    mask[1, 12:] = 0
    mask[3, 8:] = 0
    mask[5, 4:] = 0

    advantages = torch.tensor([1.0, -0.8, 0.4, -1.2, 0.2, -0.5])

    out = compute_sequence_psgppo_loss(
        per_token_logps=current_logps,
        old_per_token_logps=old_logps,
        ref_per_token_logps=ref_logps,
        advantages=advantages,
        completion_mask=mask,
        beta=0.1,
        epsilon_s=0.05,
        beta1=0.75,
        beta2=1.0,
        w_max=10.0,
        epsilon_s_mode="length_scaled",
        token_epsilon=0.2,
        epsilon_s_min=0.01,
        epsilon_s_max=0.10,
    )

    assert torch.isfinite(out.loss), f"loss is not finite: {out.loss}"
    out.loss.backward()
    assert current_logps.grad is not None, "missing gradient"
    assert torch.isfinite(current_logps.grad).all(), "gradient contains non-finite values"

    print("PS-GPPO sequence objective smoke test OK")
    print(f"loss={out.loss.item():.6f}")
    print(f"step_ratio_mean={out.step_ratio.mean().item():.6f}")
    print(f"epsilon_s_mean={out.epsilon_s.mean().item():.6f}")
    print(f"in_clip_frac={out.in_clip.float().mean().item():.3f}")
    print(f"na_lp_frac={out.na_lp.float().mean().item():.3f}")
    print(f"pa_lp_frac={out.pa_lp.float().mean().item():.3f}")
    print(f"silenced_frac={out.silenced.float().mean().item():.3f}")


if __name__ == "__main__":
    main()

