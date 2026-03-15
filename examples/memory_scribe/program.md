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
- Do NOT change the base model or dataset

## Hypotheses to Explore

1. Learning rate: try 1e-4, 5e-5, 2e-4
2. LoRA rank: 8, 16, 32
3. LoRA alpha: 16, 32, 64
4. Dropout: 0.0, 0.05, 0.1
5. Gradient accumulation steps: 4, 8, 16
6. Warmup ratio: 0.03, 0.05, 0.1
7. Weight decay: 0.0, 0.01, 0.1
8. Number of epochs: 1, 2, 3

## Success Criteria

The session is complete when val_loss ≤ 2.50 or 50 iterations are exhausted.
The best configuration should be committed with a [KEEP] tag and logged to Supabase.
