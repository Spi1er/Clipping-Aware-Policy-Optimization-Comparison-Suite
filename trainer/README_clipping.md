# MiniMind Clipping Trainer Map

The clipping comparison suite is organized around a shared training scaffold and
thin algorithm-specific entry files.

## Shared Infrastructure

- `shared_rl_utils.py`: reward shaping, completion masks, group advantages, KL terms, ratio diagnostics, moving averages, W&B schema.
- `rollout_engine.py`: old-policy snapshot rollout, old logprob computation, optional SGLang integration.
- `trainer_utils.py`: MiniMind checkpointing, model loading, reward model wrapper, W&B initialization.

## Objective Entry Points

- `train_grpo.py`: token-level PPO/GRPO clip.
- `train_dapo.py`: asymmetric lower/upper clip.
- `train_dcpo.py`: dynamic clipping bounds.
- `train_cfpo.py`: quadratic off-policy penalty.
- `train_sapo.py`: smooth adaptive gate.
- `train_gspo.py`: sequence-level geometric mean ratio.
- `train_cegppo.py`: token-level gradient-preserving CE-GPPO.
- `train_psgppo.py`: sequence-level PS-GPPO for the current RLAIF data.

## PS-GPPO Notes

The current `rlaif.jsonl` data does not contain multi-turn agent steps or process
rewards.  `train_psgppo.py` therefore treats a whole sampled completion as one
step.  This is a valid comparison against GSPO/CE-GPPO on the MiniMind setup, but
it is not yet a full agentic RL experiment.

For a quick local objective test:

```bash
python smoke_test_psgppo_sequence.py
```

For a short training sanity check:

```bash
python train_psgppo.py --max_train_steps 2 --batch_size 1 --num_generations 2 --max_gen_len 64 --num_workers 0 --wandb_mode disabled
```

For the length-calibrated variant inspired by the observed response-length and
ratio-scale issue:

```bash
python train_psgppo.py --epsilon_s_mode length_scaled --token_epsilon 0.2 --epsilon_s_min 0.01 --epsilon_s_max 0.10
```
