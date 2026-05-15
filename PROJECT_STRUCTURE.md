# Project Structure

The GitHub repository keeps the MiniMind experiment root at repository root.
The original local project may place the same files under `Adaptive Clipping/minimind/`.

## Canonical Training Files

- `trainer/shared_rl_utils.py`: reward shaping, completion masks, group advantages,
  KL terms, ratio diagnostics, moving averages, and logging schema.
- `trainer/rollout_engine.py`: frozen old-policy rollout engine and old-logprob computation.
- `trainer/train_grpo.py`: token-level GRPO baseline.
- `trainer/train_dapo.py`: asymmetric DAPO clipping.
- `trainer/train_dcpo.py`: dynamic clipping bounds.
- `trainer/train_cfpo.py`: clipping-free penalty objective.
- `trainer/train_sapo.py`: smooth adaptive penalty/gate objective.
- `trainer/train_gspo.py`: sequence-level GSPO baseline.
- `trainer/train_cegppo.py`: token-level CE-GPPO gradient-preserving baseline.
- `trainer/train_psgppo.py`: MiniMind RLAIF PS-GPPO training entry.
- `trainer/algorithms/psgppo_sequence.py`: sequence-level PS-GPPO objective helper.
- `trainer/smoke_test_psgppo_sequence.py`: CPU objective smoke test.

## Data, Models, And Outputs

- `dataset/rlaif.jsonl`: MiniMind RLAIF prompts used by the comparison suite.
- `model/`: MiniMind model definitions and tokenizer files.
- `minimind-3/`: MiniMind metadata and tokenizer/config assets.
- `reports/`: generated report assets and result tables.
- `docs/`: experiment notes, evaluation summaries, and server workflow.

Ignored local-only paths:

- `out/`, `checkpoints/`, `*.pth`, `*.safetensors`: model weights and checkpoints.
- `wandb/`, `trainer/wandb/`: W&B local run cache.
- `trainer/backup_before_wandb/`: local backup snapshot.
- `Plots/`: raw W&B exports used during exploration.
- `.DS_Store`, `.Rhistory`, `tea_debug.log`: local/generated noise.

## Smoke Tests

Objective-only smoke test:

```bash
python trainer/smoke_test_psgppo_sequence.py
```

Short PS-GPPO training sanity check:

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

Length-calibrated epsilon variant:

```bash
python trainer/train_psgppo.py \
  --from_weight full_sft \
  --save_weight psgppo_len_scaled \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --epsilon_s_mode length_scaled \
  --token_epsilon 0.2 \
  --epsilon_s_min 0.01 \
  --epsilon_s_max 0.10
```

