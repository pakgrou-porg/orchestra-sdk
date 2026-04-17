"""
synthetic/train.py — Orchestra Conductor synthetic training script
==================================================================
This script simulates a training run without any GPU, model weights, or
real dataset. It is designed for end-to-end testing of the Conductor loop
(LLM → git commit → run → metric read → keep/discard → Supabase write)
without requiring Docker, CUDA, or a real training environment.

How it works
------------
- Reads hyperparameters from this file (the Conductor will edit them).
- Simulates a noisy metric that trends downward when hyperparameters are
  within a "good" range, and trends upward (or stays flat) otherwise.
- Writes results.json in the format expected by ReadResults.
- Exits 0 on success, 1 on a simulated crash (triggered by setting
  SIMULATE_CRASH=1 in the environment, useful for testing FAILED handling).

Hyperparameters — managed by Conductor
---------------------------------------
The Conductor will propose changes to the values in the HYPERPARAMETERS
block below. Keep the block markers intact.
"""

import json
import math
import os
import random
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# HYPERPARAMETERS — managed by Conductor
# ---------------------------------------------------------------------------

LEARNING_RATE: float = 3e-4
BATCH_SIZE: int = 16
WARMUP_STEPS: int = 100
WEIGHT_DECAY: float = 0.01
DROPOUT: float = 0.1
GRADIENT_CLIP: float = 1.0
NUM_EPOCHS: int = 3
LABEL_SMOOTHING: float = 0.0

# ---------------------------------------------------------------------------
# Simulation constants (do not edit — Conductor should not touch these)
# ---------------------------------------------------------------------------

_RESULTS_FILE = Path(os.environ.get("RESULTS_FILE", "results.json"))
_SEED = int(os.environ.get("SEED", "42"))
_SIMULATE_CRASH = os.environ.get("SIMULATE_CRASH", "0") == "1"

# "Optimal" hyperparameter ranges that produce lower loss
_OPTIMAL = {
    "lr": (1e-4, 5e-4),
    "batch": (8, 32),
    "warmup": (50, 300),
    "wd": (0.005, 0.05),
    "dropout": (0.05, 0.2),
    "clip": (0.5, 2.0),
    "epochs": (2, 6),
    "label_smooth": (0.0, 0.15),
}


def _in_range(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def _compute_base_loss() -> float:
    """
    Compute a deterministic base loss from the current hyperparameters.
    Loss is lower when all hyperparameters are within their optimal ranges.
    """
    penalty = 0.0

    if not _in_range(LEARNING_RATE, *_OPTIMAL["lr"]):
        dist = min(
            abs(LEARNING_RATE - _OPTIMAL["lr"][0]),
            abs(LEARNING_RATE - _OPTIMAL["lr"][1]),
        )
        penalty += math.log10(max(dist / 1e-4, 1)) * 0.3

    if not _in_range(BATCH_SIZE, *_OPTIMAL["batch"]):
        penalty += 0.15

    if not _in_range(WARMUP_STEPS, *_OPTIMAL["warmup"]):
        penalty += 0.10

    if not _in_range(WEIGHT_DECAY, *_OPTIMAL["wd"]):
        penalty += 0.10

    if not _in_range(DROPOUT, *_OPTIMAL["dropout"]):
        penalty += 0.12

    if not _in_range(GRADIENT_CLIP, *_OPTIMAL["clip"]):
        penalty += 0.08

    if not _in_range(NUM_EPOCHS, *_OPTIMAL["epochs"]):
        penalty += 0.05

    if not _in_range(LABEL_SMOOTHING, *_OPTIMAL["label_smooth"]):
        penalty += 0.07

    # Base loss starts at 2.40; optimal config converges toward ~2.10
    return 2.40 + penalty


def _simulate_training(base_loss: float) -> dict:
    """
    Simulate epoch-by-epoch training with noise.
    Returns final metrics dict.
    """
    rng = random.Random(_SEED + int(LEARNING_RATE * 1e6) + BATCH_SIZE)
    loss = base_loss + rng.uniform(0.05, 0.15)  # starting loss with noise

    epoch_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        # Each epoch reduces loss by a noisy amount
        improvement = rng.uniform(0.05, 0.14) * (1.0 - 0.08 * (epoch - 1))
        noise = rng.gauss(0, 0.01)
        loss = max(loss - improvement + noise, 1.5)  # floor at 1.5
        epoch_losses.append(round(loss, 4))
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  loss={loss:.4f}", flush=True)
        time.sleep(0.2)  # simulate compute time

    val_loss = epoch_losses[-1] + rng.gauss(0, 0.008)
    val_accuracy = max(0.0, min(1.0, 0.95 - val_loss * 0.15 + rng.gauss(0, 0.005)))

    return {
        "val_loss": round(val_loss, 4),
        "val_accuracy": round(val_accuracy, 4),
        "train_loss": round(epoch_losses[-1], 4),
        "epoch": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "gradient_clip": GRADIENT_CLIP,
        "label_smoothing": LABEL_SMOOTHING,
        "epoch_losses": epoch_losses,
    }


def main() -> None:
    print("=" * 60)
    print("Orchestra Synthetic Training Script")
    print("=" * 60)
    print(f"  learning_rate      = {LEARNING_RATE}")
    print(f"  batch_size         = {BATCH_SIZE}")
    print(f"  warmup_steps       = {WARMUP_STEPS}")
    print(f"  weight_decay       = {WEIGHT_DECAY}")
    print(f"  dropout            = {DROPOUT}")
    print(f"  gradient_clip      = {GRADIENT_CLIP}")
    print(f"  num_epochs         = {NUM_EPOCHS}")
    print(f"  label_smoothing    = {LABEL_SMOOTHING}")
    print()

    if _SIMULATE_CRASH:
        print("SIMULATE_CRASH=1 — raising RuntimeError to test FAILED handling")
        raise RuntimeError("Simulated OOM: CUDA out of memory (synthetic)")

    base_loss = _compute_base_loss()
    print(f"Computed base loss from hyperparameters: {base_loss:.4f}")
    print()

    metrics = _simulate_training(base_loss)

    print()
    print(f"Training complete.")
    print(f"  val_loss     = {metrics['val_loss']}")
    print(f"  val_accuracy = {metrics['val_accuracy']}")
    print(f"  train_loss   = {metrics['train_loss']}")

    _RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults written to {_RESULTS_FILE}")


if __name__ == "__main__":
    main()
