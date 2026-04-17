# Synthetic Hyperparameter Optimisation — Conductor Program

## Objective

Minimise `val_loss` in `train.py` by iteratively adjusting hyperparameters.
The target is `val_loss ≤ 2.30`. The script simulates a real training run
without requiring a GPU or model weights.

## Training Script

`train.py` exposes the following hyperparameters in the `HYPERPARAMETERS` block:

| Parameter | Type | Current default | Notes |
|---|---|---|---|
| `LEARNING_RATE` | float | `3e-4` | Try range 1e-5 to 1e-3 |
| `BATCH_SIZE` | int | `16` | Powers of 2 preferred |
| `WARMUP_STEPS` | int | `100` | Typically 5–10% of total steps |
| `WEIGHT_DECAY` | float | `0.01` | L2 regularisation |
| `DROPOUT` | float | `0.1` | Applied to attention layers |
| `GRADIENT_CLIP` | float | `1.0` | Max gradient norm |
| `NUM_EPOCHS` | int | `3` | More epochs = slower but lower loss |
| `LABEL_SMOOTHING` | float | `0.0` | Cross-entropy label smoothing |

## Rules

1. Only edit values inside the `HYPERPARAMETERS` block. Do not touch the simulation constants or the `main()` function.
2. Change at most **2–3 hyperparameters per iteration** so the effect of each change is interpretable.
3. Always provide a hypothesis explaining *why* the proposed change should reduce `val_loss`.
4. If a change is discarded, do not repeat it in the next iteration. Try a different direction.
5. Use memory of past experiments to avoid revisiting configurations that were already tried.

## Success Criteria

The session is complete when `val_loss ≤ 2.30` is achieved and the result is committed.
