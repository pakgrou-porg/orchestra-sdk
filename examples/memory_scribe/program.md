# Research Program: memory_scribe_v1

## Goal

Minimize `val_loss` on the `memory_scribe_v1` dataset — a question-answering dataset
focused on extracting and summarizing information from context passages.

## Current Baseline

- Model: Qwen 3.5 7B (quantized, loaded via Unsloth)
- Dataset: memory_scribe_v1 (80 train / 20 eval)
- Current val_loss: ~2.85 (untrained baseline)
- Target val_loss: ≤ 2.50

## Approach

Fine-tune using LoRA adapters with the Unsloth library for 4-bit quantized training.
The training loop uses AdamW with cosine LR scheduling.

## Constraints

- Max training time per iteration: 45 minutes
- GPU: NVIDIA RTX 4060 (8GB VRAM)
- Batch size must keep VRAM usage under 7GB (use gradient accumulation if needed)
- LoRA rank: 8–64 (start at 16)
- USE_BF16: controls whether bfloat16 (True) or float16 (False) is used for training.
  The RTX 4060 is Ada Lovelace architecture (Ampere+), so USE_BF16 = True is correct.
  Only change USE_BF16 if the target GPU changes — it is a hardware compatibility flag,
  not a training quality hyperparameter. Do not toggle it between runs on the same machine.
- Do NOT change the base model or dataset

## Hyperparameters Managed by the Conductor

The following variables in `train.py` may be modified each iteration.
All are in the `# Hyperparameters — managed by Conductor` block at the top of the file.

| Variable | Type | Current | Allowed range / values |
|----------|------|---------|------------------------|
| `LEARNING_RATE` | float | 2e-4 | 5e-5 – 5e-4 |
| `BATCH_SIZE` | int | 2 | 1, 2, 4 |
| `GRADIENT_ACCUMULATION_STEPS` | int | 4 | 4, 8, 16 |
| `NUM_EPOCHS` | int | 1 | 1, 2, 3 |
| `WARMUP_RATIO` | float | 0.05 | 0.03, 0.05, 0.1 |
| `WEIGHT_DECAY` | float | 0.01 | 0.0, 0.01, 0.1 |
| `MAX_SEQ_LENGTH` | int | 2048 | 1024, 2048 |
| `LORA_RANK` | int | 16 | 8, 16, 32, 64 |
| `LORA_ALPHA` | int | 32 | 16, 32, 64 |
| `LORA_DROPOUT` | float | 0.05 | 0.0, 0.05, 0.1 |
| `USE_BF16` | bool | True | True (Ampere+), False (pre-Ampere) |

## Hypotheses to Explore

1. Learning rate: try 1e-4, 5e-5, 2e-4
2. LoRA rank: 8, 16, 32
3. LoRA alpha: 16, 32, 64
4. Dropout: 0.0, 0.05, 0.1
5. Gradient accumulation steps: 4, 8, 16
6. Warmup ratio: 0.03, 0.05, 0.1
7. Weight decay: 0.0, 0.01, 0.1
8. Number of epochs: 1, 2, 3
9. USE_BF16: confirm True is stable on this GPU; switch to False only if OOM or NaN loss occurs

## Success Criteria

The session is complete when val_loss ≤ 2.50 or 50 iterations are exhausted.
The best configuration should be committed with a [KEEP] tag and logged to Supabase.
