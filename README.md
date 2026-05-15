# Clipping-Aware Policy Optimization Comparison Suite

This repository is a MiniMind-based research codebase for studying clipping-aware
policy optimization in LLM post-training. It implements and compares PPO/GRPO-style
objectives, adaptive clipping variants, and a sequence-level **PS-GPPO**
(Per-Step Gradient-Preserving Policy Optimization) objective.

The current experiments focus on reasoning-oriented RLAIF training with a 768-hidden
MiniMind model. The main question is how clipping granularity and gradient preservation
affect reward improvement, formatting stability, repetition, and max-length truncation.

## Highlights

- Unified trainer scaffold for GRPO, DAPO, DCPO, CFPO, SAPO, GSPO, CE-GPPO, and PS-GPPO.
- Frozen old-policy rollout engine and shared reward / KL / ratio diagnostics.
- W&B logging for reward, KL, entropy, clipped fraction, preserved-gradient fraction,
  step ratio, and PS-GPPO region statistics.
- Fixed-prompt offline evaluation script for comparing SFT, GRPO, CE-GPPO, and PS-GPPO.
- Reproducible smoke tests and server training workflow.

## Current Result Snapshot

All rows below use the same 64-prompt offline evaluation set unless noted otherwise.

| Method | Reward mean | Pairwise win rate | Max-length hit | Exactly one `</think>` | Repetition penalty |
|---|---:|---:|---:|---:|---:|
| full-SFT | -1.4686 | - | 64.1% | 62.5% | 0.0552 |
| GRPO 2k | -0.6457 | 90.6% vs SFT | 48.4% | 71.9% | 0.0405 |
| CE-GPPO 2k | -0.7176 | 78.1% vs SFT | 42.2% | 76.6% | 0.0315 |
| PS-GPPO 2k | 0.6938 | 96.9% vs SFT | 31.3% | 84.4% | 0.0271 |

Pairwise comparisons on the same prompts:

- PS-GPPO vs GRPO: 87.5% win rate, +1.34 average reward.
- PS-GPPO vs CE-GPPO: 90.6% win rate, +1.41 average reward.

The main observed improvement is better reward-model preference, more stable
`</think>` closure, lower repetition, and fewer truncated generations. This should
not yet be interpreted as a full multi-turn Agentic RL result: the current MiniMind
RLAIF dataset is single-turn, so PS-GPPO is evaluated as a sequence-level bridge
experiment where one sampled completion is treated as one step.

## Repository Structure

```text
.
├── trainer/
│   ├── train_grpo.py              # token-level GRPO baseline
│   ├── train_cegppo.py            # token-level gradient-preserving baseline
│   ├── train_gspo.py              # sequence-level GSPO baseline
│   ├── train_psgppo.py            # sequence-level PS-GPPO entry point
│   ├── algorithms/
│   │   └── psgppo_sequence.py     # PS-GPPO objective helper
│   ├── rollout_engine.py          # old-policy rollout and logprob computation
│   ├── shared_rl_utils.py         # reward, masks, advantages, KL, diagnostics
│   └── smoke_test_psgppo_sequence.py
├── dataset/
│   └── rlaif.jsonl                # MiniMind RLAIF prompts
├── model/                         # MiniMind model definition and tokenizer files
├── scripts/
│   └── eval_psgppo_compare.py     # fixed-prompt offline evaluation
├── docs/
│   ├── evaluation_results.md
│   └── training_workflow.md
├── reports/
│   └── assets/summary_table.csv
├── PROJECT_STRUCTURE.md
└── PS_GPPO.md
```

Local checkpoints, model weights, W&B runs, and backup snapshots are intentionally
ignored by Git. Keep them locally or upload them as release assets if needed.

## Quick Start

Install dependencies from the repository root:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Run the objective-only PS-GPPO smoke test:

```bash
python trainer/smoke_test_psgppo_sequence.py
```

Run a short training sanity check after preparing a local reward model:

```bash
python trainer/train_psgppo.py \
  --from_weight full_sft \
  --save_weight psgppo_smoke \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --batch_size 1 \
  --num_generations 2 \
  --num_workers 0 \
  --max_gen_len 64 \
  --max_train_steps 2 \
  --save_interval 100000 \
  --wandb_mode disabled
```

Run fixed-prompt evaluation:

```bash
python scripts/eval_psgppo_compare.py \
  --root "$(pwd)" \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --weights full_sft grpo_baseline_2k ce_gppo_2k psgppo_run1 \
  --num_samples 64
```

## Key Experiment Runs

- PS-GPPO 2k W&B run: `rmmtoboa`
- GRPO 2k W&B run: `neiqqi3j`
- CE-GPPO 2k W&B run: `o47sp0zf`

## Notes

This repository is a research and comparison suite, not a production post-training
pipeline. The strongest current evidence supports improvements in reward preference,
format stability, length control, and repetition reduction under the MiniMind/RLAIF
setup. Multi-step Agentic RL with real tool-use trajectories remains the next stage.
