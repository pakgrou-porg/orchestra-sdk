"""
Unit tests for orchestra_sdk tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestra_sdk.tools.file_tools import EditFile, EditFileError, FileToolError, ListFiles, ReadFile, WriteFile
from orchestra_sdk.tools.git_tools import GitManager, GitCommitError
from orchestra_sdk.tools.run_experiment import ReadResults, ResultsNotFoundError, ResultsParseError


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_existing_file(self, workspace: Path):
        reader = ReadFile(workspace)
        content = reader.run("train.py")
        assert "LEARNING_RATE" in content
        assert "BATCH_SIZE" in content

    def test_raises_on_missing_file(self, workspace: Path):
        reader = ReadFile(workspace)
        with pytest.raises(FileNotFoundError):
            reader.run("nonexistent.py")

    def test_raises_on_path_traversal(self, workspace: Path):
        reader = ReadFile(workspace)
        with pytest.raises((PermissionError, FileToolError)):
            reader.run("../../etc/passwd")


class TestEditFile:
    def test_simple_edit(self, workspace: Path):
        editor = EditFile(workspace)
        result = editor.run(
            path="train.py",
            find="LEARNING_RATE = 0.01",
            replace="LEARNING_RATE = 0.001",
        )
        assert result["applied"] is True
        content = (workspace / "train.py").read_text()
        assert "LEARNING_RATE = 0.001" in content
        assert "LEARNING_RATE = 0.01\n" not in content

    def test_raises_on_missing_find_string(self, workspace: Path):
        editor = EditFile(workspace)
        with pytest.raises(EditFileError, match="not found"):
            editor.run(
                path="train.py",
                find="NONEXISTENT_STRING_XYZ",
                replace="something",
            )

    def test_raises_on_multiple_matches_when_strict(self, workspace: Path):
        # Write a file with duplicate strings
        (workspace / "dup.py").write_text("x = 1\nx = 1\n")
        editor = EditFile(workspace)
        with pytest.raises(EditFileError, match="matched"):
            editor.run(
                path="dup.py",
                find="x = 1",
                replace="x = 2",
                validate_single_match=True,
            )

    def test_allows_multiple_matches_when_not_strict(self, workspace: Path):
        (workspace / "dup.py").write_text("x = 1\nx = 1\n")
        editor = EditFile(workspace)
        result = editor.run(
            path="dup.py",
            find="x = 1",
            replace="x = 2",
            validate_single_match=False,
        )
        assert result["applied"] is True
        content = (workspace / "dup.py").read_text()
        assert content == "x = 2\nx = 2\n"


class TestListFiles:
    def test_lists_python_files(self, workspace: Path):
        lister = ListFiles(workspace)
        result = lister.run(pattern="*.py")
        # result is a list of dicts with 'path' key
        paths = [r["path"] for r in result]
        assert any("train.py" in p for p in paths)
        assert any("evaluate.py" in p for p in paths)

    def test_lists_all_files(self, workspace: Path):
        lister = ListFiles(workspace)
        result = lister.run()
        assert len(result) >= 3  # train.py, evaluate.py, program.md


# ---------------------------------------------------------------------------
# Git tools
# ---------------------------------------------------------------------------


class TestGitManager:
    def test_initialize_creates_repo(self, workspace: Path):
        git = GitManager(workspace, branch="test/session")
        git.initialize()
        assert (workspace / ".git").exists()

    def test_commit_creates_commit(self, workspace: Path):
        git = GitManager(workspace, branch="test/session")
        git.initialize()
        result = git.commit.run("Initial commit")
        assert "sha" in result
        assert len(result["sha"]) == 40

    def test_log_returns_commits(self, workspace: Path):
        git = GitManager(workspace, branch="test/session")
        git.initialize()
        git.commit.run("First commit")
        (workspace / "train.py").write_text("# modified\n")
        git.commit.run("Second commit")
        log = git.log.run(n=5)
        assert len(log) >= 2

    def test_reset_reverts_to_sha(self, workspace: Path):
        git = GitManager(workspace, branch="test/session")
        git.initialize()
        result1 = git.commit.run("Baseline commit")
        sha1 = result1["sha"]

        # Make a change and commit
        (workspace / "train.py").write_text("LEARNING_RATE = 0.001\n")
        git.commit.run("Candidate commit")

        # Reset to baseline
        git.reset.run(sha1)
        content = (workspace / "train.py").read_text()
        assert "LEARNING_RATE = 0.001" not in content


# ---------------------------------------------------------------------------
# ReadResults
# ---------------------------------------------------------------------------


class TestReadResults:
    def test_reads_valid_results(self, workspace: Path, results_json: Path, minimal_config_dict: dict, set_test_env):
        from orchestra_sdk.config import ConductorConfig
        config = ConductorConfig.model_validate(minimal_config_dict)
        reader = ReadResults(config)
        result = reader.run()
        assert result["target_metric"] == pytest.approx(2.847)
        assert result["target_metric_name"] == "val_loss"
        assert result["epoch"] == 5

    def test_raises_on_missing_results(self, workspace: Path, minimal_config_dict: dict, set_test_env):
        from orchestra_sdk.config import ConductorConfig
        config = ConductorConfig.model_validate(minimal_config_dict)
        reader = ReadResults(config)
        with pytest.raises(ResultsNotFoundError):
            reader.run()

    def test_raises_on_invalid_json(self, workspace: Path, minimal_config_dict: dict, set_test_env):
        (workspace / "results.json").write_text("not json {{{")
        from orchestra_sdk.config import ConductorConfig
        config = ConductorConfig.model_validate(minimal_config_dict)
        reader = ReadResults(config)
        with pytest.raises(ResultsParseError):
            reader.run()

    def test_raises_on_missing_target_metric(self, workspace: Path, minimal_config_dict: dict, set_test_env):
        (workspace / "results.json").write_text(json.dumps({"train_loss": 1.5}))
        from orchestra_sdk.config import ConductorConfig
        config = ConductorConfig.model_validate(minimal_config_dict)
        reader = ReadResults(config)
        with pytest.raises(ResultsParseError, match="val_loss"):
            reader.run()
