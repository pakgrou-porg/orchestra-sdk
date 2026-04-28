"""
memory_scribe_v1 — LoRA fine-tuning script (Unsloth native)
============================================================
Edited iteratively by the Orchestra Conductor.
DO NOT manually edit hyperparameter values — they are managed by the Conductor.

Uses Unsloth's FastLanguageModel + TRL SFTTrainer for ~2× faster training
and ~60% lower VRAM usage compared to vanilla PEFT + HuggingFace Trainer.

Dataset format (JSONL, one record per line):
    {"system_prompt": "...", "user_prompt": "...", "assistant_response": "..."}

Each record is reshaped into the ShareGPT conversations format that Unsloth's
apply_chat_template() understands natively — no manual tokenisation needed.

To run manually:
    python train.py

Requirements (included in orchestra-musician Docker image):
    pip install "unsloth[colab-new]" trl transformers datasets peft accelerate bitsandbytes
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Hyperparameters — managed by Conductor
# ---------------------------------------------------------------------------

LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 1
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_SEQ_LENGTH = 2048

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Precision — set True for Ampere+ GPUs (RTX 30xx / A100 / H100)
# set False for pre-Ampere GPUs (RTX 20xx or older) which require fp16
USE_BF16 = True

# ---------------------------------------------------------------------------
# Paths and model
# ---------------------------------------------------------------------------

DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", "/datasets"))
DATASET_NAME = "memory_scribe_v1"
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
OUTPUT_DIR = Path("./output")
RESULTS_FILE = Path("results.json")

# ---------------------------------------------------------------------------
# Data loading and ShareGPT reshape
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_sharegpt(records: list[dict]) -> list[dict]:
    """
    Reshape Orchestra dataset records into the ShareGPT conversations format.

    Input:  {"system_prompt": str, "user_prompt": str, "assistant_response": str}
    Output: {"conversations": [{"role": "system",    "value": ...},
                                {"role": "user",      "value": ...},
                                {"role": "assistant", "value": ...}]}

    Unsloth's get_chat_template() and apply_chat_template() consume this
    format directly — no manual string formatting or tokenisation required.
    """
    result = []
    for r in records:
        turns = []
        system = r.get("system_prompt", "").strip()
        if system:
            turns.append({"role": "system", "value": system})
        turns.append({"role": "user",      "value": r.get("user_prompt", "").strip()})
        turns.append({"role": "assistant", "value": r.get("assistant_response", "").strip()})
        result.append({"conversations": turns})
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main() -> None:
    start_time = time.time()

    iteration = int(os.environ.get("ITERATION", 0))
    sha = os.environ.get("HYPOTHESIS_SHA", "unknown")
    print(f"[Orchestra] iteration={iteration} sha={sha[:8]}")

    # ── 1. Load model and tokeniser via Unsloth ──────────────────────────────
    from unsloth import FastLanguageModel  # type: ignore

    print(f"[Orchestra] Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # auto-detect: bfloat16 on Ampere+, float16 otherwise
    )

    # ── 2. Apply LoRA via Unsloth (fused kernels, gradient checkpointing) ────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves VRAM on long sequences
        random_state=42,
        use_rslora=False,
    )
    model.print_trainable_parameters()

    # ── 3. Load dataset and reshape to ShareGPT format ───────────────────────
    train_path = DATASETS_DIR / DATASET_NAME / "train.jsonl"
    eval_path  = DATASETS_DIR / DATASET_NAME / "eval.jsonl"

    print(f"[Orchestra] Loading dataset from {train_path}")
    train_records = load_jsonl(train_path)
    eval_records  = load_jsonl(eval_path)

    from datasets import Dataset  # type: ignore

    train_dataset = Dataset.from_list(to_sharegpt(train_records))
    eval_dataset  = Dataset.from_list(to_sharegpt(eval_records))
    print(f"[Orchestra] train={len(train_dataset)} eval={len(eval_dataset)} samples")

    # ── 4. Apply chat template via Unsloth ───────────────────────────────────
    from unsloth.chat_templates import get_chat_template  # type: ignore

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    def apply_template(batch: dict) -> dict:
        texts = tokenizer.apply_chat_template(
            batch["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": texts}

    train_dataset = train_dataset.map(
        apply_template, batched=True, remove_columns=["conversations"]
    )
    eval_dataset = eval_dataset.map(
        apply_template, batched=True, remove_columns=["conversations"]
    )

    # ── 5. SFTTrainer ─────────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig  # type: ignore

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,       # set True for short sequences to improve throughput
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("[Orchestra] Starting training …")
    trainer.train()

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    print("[Orchestra] Evaluating …")
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss", 99.0)

    # ── 8. Write results.json — REQUIRED by the Conductor ────────────────────
    duration = time.time() - start_time
    log_history = trainer.state.log_history
    train_loss = next(
        (e["loss"] for e in reversed(log_history) if "loss" in e), 0.0
    )

    results = {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "epoch": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "model": MODEL_NAME,
        "use_bf16": USE_BF16,
        "iteration": iteration,
        "hypothesis_sha": sha,
        "duration_seconds": round(duration, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Orchestra] Results written to {RESULTS_FILE}")
    print(f"[Orchestra] val_loss={val_loss:.4f} | duration={duration:.1f}s")


if __name__ == "__main__":
    main()
