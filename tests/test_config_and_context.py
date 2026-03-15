"""
Tests for config loading, validation, and context assembly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from orchestra_sdk.config import ConductorConfig
from orchestra_sdk.context import assemble_context, Hypothesis


class TestConductorConfig:
    def test_loads_from_dict(self, minimal_config_dict: dict, set_test_env):
        config = ConductorConfig.model_validate(minimal_config_dict)
        assert config.session.name == "test_session"
        assert config.session.target_metric == "val_loss"
        assert config.session.keep_threshold == -0.005

    def test_loads_from_yaml(self, workspace: Path, minimal_config_dict: dict, set_test_env):
        yaml_path = workspace / "conductor_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(minimal_config_dict, f)
        config = ConductorConfig.from_yaml(str(yaml_path))
        assert config.session.name == "test_session"

    def test_workspace_path_resolves(self, workspace: Path, minimal_config_dict: dict, set_test_env):
        config = ConductorConfig.model_validate(minimal_config_dict)
        assert config.session.workspace_path == workspace

    def test_api_key_from_env(self, minimal_config_dict: dict, set_test_env):
        config = ConductorConfig.model_validate(minimal_config_dict)
        assert config.llm.get_api_key() == "sk-test-key-12345"

    def test_missing_api_key_env_raises(self, minimal_config_dict: dict):
        # Don't set the env var — should raise OSError
        import os
        os.environ.pop("TEST_API_KEY", None)
        config = ConductorConfig.model_validate(minimal_config_dict)
        with pytest.raises((OSError, KeyError, ValueError)):
            config.llm.get_api_key()

    def test_invalid_keep_threshold_raises(self, minimal_config_dict: dict, set_test_env):
        minimal_config_dict["session"]["keep_threshold"] = 0.1  # positive — invalid
        with pytest.raises(Exception):
            ConductorConfig.model_validate(minimal_config_dict)

    def test_default_values(self, minimal_config_dict: dict, set_test_env):
        config = ConductorConfig.model_validate(minimal_config_dict)
        assert config.llm.max_tokens == 4096
        assert config.llm.temperature == 0.3
        assert config.memory.top_k == 5
        assert config.memory.similarity_threshold == 0.75


class TestContextAssembly:
    def test_assembles_messages(self, workspace: Path, minimal_config_dict: dict, set_test_env):
        config = ConductorConfig.model_validate(minimal_config_dict)
        messages = assemble_context(
            config=config,
            program_text="# Research Program\nGoal: minimize val_loss.",
            train_script_text="LEARNING_RATE = 0.01\nBATCH_SIZE = 32\n",
            git_log=[],
            metric_history=[],
            memories=[],
            baseline_metric=2.847,
            iteration=1,
        )
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "LEARNING_RATE" in messages[1].content
        assert "val_loss" in messages[1].content
        assert "2.8470" in messages[1].content

    def test_truncates_large_train_script(self, workspace: Path, minimal_config_dict: dict, set_test_env):
        config = ConductorConfig.model_validate(minimal_config_dict)
        # Create a very large train.py
        large_script = "x = 1\n" * 10000
        messages = assemble_context(
            config=config,
            program_text="# Program",
            train_script_text=large_script,
            git_log=[],
            metric_history=[],
            memories=[],
            baseline_metric=None,
            iteration=1,
        )
        # Should not raise and should be within budget
        total_chars = sum(len(m.content) for m in messages)
        # Budget is 8000 tokens * 4 chars = 32000 chars; add system prompt
        assert total_chars < 50000  # Generous upper bound

    def test_first_iteration_shows_no_baseline(self, workspace: Path, minimal_config_dict: dict, set_test_env):
        config = ConductorConfig.model_validate(minimal_config_dict)
        messages = assemble_context(
            config=config,
            program_text="# Program",
            train_script_text="LR = 0.01\n",
            git_log=[],
            metric_history=[],
            memories=[],
            baseline_metric=None,
            iteration=1,
        )
        assert "first iteration" in messages[1].content.lower() or "N/A" in messages[1].content


class TestHypothesisSchema:
    def test_valid_hypothesis(self):
        h = Hypothesis(
            hypothesis="Reduce learning rate from 0.01 to 0.001",
            change_type="hyperparameter",
            expected_effect="reduce val_loss by ~0.05",
            risk="low",
            confidence=0.75,
            memory_note="Tried reducing LR; expected improvement.",
            edit={
                "find": "LEARNING_RATE = 0.01",
                "replace": "LEARNING_RATE = 0.001",
            },
        )
        assert h.confidence == 0.75
        assert h.edit.find == "LEARNING_RATE = 0.01"

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(Exception):
            Hypothesis(
                hypothesis="test",
                change_type="hyperparameter",
                expected_effect="test",
                risk="low",
                confidence=1.5,  # > 1.0
                memory_note="test",
                edit={"find": "x", "replace": "y"},
            )
