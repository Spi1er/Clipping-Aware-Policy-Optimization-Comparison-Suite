# Evaluation Results

## Fixed-Prompt Offline Evaluation

All compared checkpoints were evaluated on the same 64 sampled prompts with the same
MiniMind model size and reward model scoring pipeline.

| Method | Reward mean | Reward min | Reward max | Pairwise win | Max-length hit | Has `</think>` | Exactly one `</think>` | Repetition penalty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full-SFT | -1.4686 | -2.0 | 1.0 | - | 64.1% | 62.5% | 62.5% | 0.0552 |
| GRPO 2k | -0.6457 | -2.0 | 2.0 | 90.6% vs SFT | 48.4% | 71.9% | 71.9% | 0.0405 |
| CE-GPPO 2k | -0.7176 | -2.0 | 2.0 | 78.1% vs SFT | 42.2% | 76.6% | 76.6% | 0.0315 |
| PS-GPPO 2k | 0.6938 | -2.0 | 2.0 | 96.9% vs SFT | 31.3% | 84.4% | 84.4% | 0.0271 |

Pairwise comparisons:

| Comparison | Win rate | Avg reward delta |
|---|---:|---:|
| PS-GPPO over SFT | 96.9% | +2.16 |
| PS-GPPO over GRPO | 87.5% | +1.34 |
| PS-GPPO over CE-GPPO | 90.6% | +1.41 |
| GRPO over SFT | 90.6% | +0.82 |
| CE-GPPO over SFT | 78.1% | +0.75 |
| CE-GPPO over GRPO | 46.9% | -0.07 |

## Training Curve Summary

- PS-GPPO final 100-step training reward mean was about `0.94`, improving from the
  first run's late-stage value around `-0.70`.
- GRPO and CE-GPPO final 100-step training reward means were around `-0.65` and
  `-0.67`, respectively.
- PS-GPPO kept step ratio near `1.00`, in-clip fraction around `0.65`, and silenced
  fraction close to `0`.
- GRPO token-level clipped fraction averaged about `0.010`.
- CE-GPPO preserved-gradient fraction averaged about `0.008`.

## Interpretation

The current evidence supports the following claims:

- PS-GPPO substantially improves reward-model preference over full-SFT, GRPO, and
  CE-GPPO under the current MiniMind/RLAIF setup.
- Improvements are strongly tied to output-format stability, reduced truncation, and
  lower repetition.
- Token-level out-of-clip regions were rarely activated in this setting, which helps
  explain why CE-GPPO's gradient-preserving token objective did not outperform GRPO.
- The MiniMind RLAIF data is single-turn, so this should be described as a
  sequence-level bridge experiment rather than a full multi-turn Agentic RL result.

## Remaining Risks

- Reward hacking is still possible: the model may learn shorter, cleaner, more
  reward-model-friendly outputs without reliably improving verifiable task correctness.
- The strongest result is not yet a full matched sweep over all older objectives
  such as DAPO, DCPO, CFPO, SAPO, and GSPO under the exact same 2k-step protocol.
- A more mature next stage should add rule-based verifiable rewards, tool-call
  validity checks, larger fixed eval sets, bootstrap confidence intervals, and
  multi-step trajectory data.

