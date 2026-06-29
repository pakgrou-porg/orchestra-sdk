"""
Shared test fixtures for orchestra_sdk tests.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# ---------------------------------------------------------------------------
# Workspace fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A temporary workspace directory with a minimal train.py and program.md."""
    # Create program.md
    (tmp_path / "program.md").write_text(
        "# Test Research Program\n\n"
        "Goal: Minimize val_loss on the test dataset.\n"
        "Current approach: simple MLP with SGD optimizer.\n"
    )

    # Create a minimal train.py
    (tmp_path / "train.py").write_text(
        "# Orchestra test train.py\n"
        "import json\n\n"
        "LEARNING_RATE = 0.01\n"
        "BATCH_SIZE = 32\n"
        "EPOCHS = 5\n"
        "DROPOUT = 0.1\n\n"
        "def train():\n"
        "    # Simulate training\n"
        "    val_loss = 3.0 - (LEARNING_RATE * 10)\n"
        "    results = {\n"
        "        'val_loss': val_loss,\n"
        "        'train_loss': val_loss - 0.3,\n"
        "        'epoch': EPOCHS,\n"
        "    }\n"
        "    with open('results.json', 'w') as f:\n"
        "        json.dump(results, f)\n"
        "    print(f'val_loss={val_loss:.4f}')\n\n"
        "if __name__ == '__main__':\n"
        "    train()\n"
    )

    # Create evaluate.py (read-only harness)
    (tmp_path / "evaluate.py").write_text(
        "# Evaluation harness — read-only, never edited by Conductor\n"
        "import json\n\n"
        "def evaluate():\n"
        "    with open('results.json') as f:\n"
        "        return json.load(f)\n\n"
        "if __name__ == '__main__':\n"
        "    print(evaluate())\n"
    )

    return tmp_path


@pytest.fixture
def results_json(workspace: Path) -> Path:
    """Write a sample results.json to the workspace."""
    results = {
        "val_loss": 2.847,
        "train_loss": 2.103,
        "val_accuracy": 0.612,
        "epoch": 5,
    }
    path = workspace / "results.json"
    path.write_text(json.dumps(results))
    return path


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config_dict(workspace: Path) -> dict:
    """A minimal valid config dict for testing."""
    return {
        "session": {
            "name": "test_session",
            "dataset_id": "test_dataset",
            "workspace_dir": str(workspace),
            "branch": "session/test",
            "max_iterations": 5,
            "target_metric": "val_loss",
            "keep_threshold": -0.005,
        },
        "llm": {
            "provider": "openrouter",
            "model": "anthropic/claude-3-7-sonnet-20250219",
            "api_key_env": "TEST_API_KEY",
        },
        "runner": {
            "type": "docker",
            "image": "orchestra-musician:latest",
            "gpu_device": "none",
            "timeout_seconds": 300,
        },
        "memory": {
            "enabled": False,  # Disable for unit tests
        },
        "supabase": {
            "url_env": "TEST_SUPABASE_URL",
            "key_env": "TEST_SUPABASE_KEY",
        },
    }


@pytest.fixture
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test-key-12345")
    monkeypatch.setenv("TEST_SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("TEST_SUPABASE_KEY", "test-anon-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-12345")
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    # SupabaseConfig now defaults to the service-role key (trusted backend) with
    # the anon key as a fallback, so set both for tests.
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "test-anon-key")
