"""
memory_scribe_v1 — LoRA fine-tuning script
==========================================
Edited iteratively by the Orchestra Conductor.
DO NOT manually edit hyperparameter values — they are managed by the Conductor.

To run manually:
    python train.py

Requirements:
    pip install unsloth transformers datasets torch peft
    (or use the orchestra-musician Docker image)
"""

import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Hyperparameters — managed by Conductor
# ---------------------------------------------------------------------------

LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_SEQ_LENGTH = 2048

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", "/datasets"))
DATASET_NAME = "memory_scribe_v1"
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
OUTPUT_DIR = Path("./output")
RESULTS_FILE = Path("results.json")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset_from_jsonl(path: Path) -> Dataset:
    """Load a JSONL dataset file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_sample(sample: dict) -> str:
    """Format a sample as a chat conversation."""
    system = sample.get("system_prompt", "You are a helpful assistant.")
    user = sample.get("user_prompt", "")
    assistant = sample.get("assistant_response", "")
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    start_time = time.time()

    # Load tokenizer and model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    train_path = DATASETS_DIR / DATASET_NAME / "train.jsonl"
    eval_path = DATASETS_DIR / DATASET_NAME / "eval.jsonl"

    print(f"Loading dataset from {train_path}")
    train_dataset = load_dataset_from_jsonl(train_path)
    eval_dataset = load_dataset_from_jsonl(eval_path)

    # Tokenize
    def tokenize(sample):
        text = format_sample(sample)
        tokens = tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_tokenized = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    eval_tokenized = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        fp16=True,
        dataloader_num_workers=2,
        report_to="none",
    )

    # Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Evaluate
    print("Evaluating...")
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss", 99.0)

    # Write results.json — REQUIRED by Conductor
    duration = time.time() - start_time
    results = {
        "val_loss": val_loss,
        "train_loss": trainer.state.log_history[-1].get("loss", 0.0),
        "epoch": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "duration_seconds": duration,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_FILE}")
    print(f"val_loss={val_loss:.4f} | duration={duration:.1f}s")


if __name__ == "__main__":
    main()
