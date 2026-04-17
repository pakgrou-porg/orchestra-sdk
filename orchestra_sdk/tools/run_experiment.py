"""
orchestra_sdk.tools.run_experiment
====================================
Tools for launching training runs and reading results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .base import BaseTool, ToolError
from ..config import ConductorConfig
from ..runner.docker_runner import (
    DockerRunner,
    ExperimentFailedError,
    ExperimentTimeoutError,
)


class RunExperimentError(ToolError):
    pass


class ResultsNotFoundError(ToolError):
    pass


class ResultsParseError(ToolError):
    pass


class RunExperiment(BaseTool):
    name = "run_experiment"
    description = (
        "Launch a training run for the current iteration and wait for completion. "
        "Returns exit code, duration, and the last 50 lines of training log."
    )

    def __init__(self, config: ConductorConfig):
        self.config = config
        workspace = config.session.workspace_path
        datasets = config.program.datasets_path

        if config.runner.type == "docker":
            self._runner = DockerRunner(config.runner, workspace, datasets)
        elif config.runner.type == "k8s":
            from ..runner.k8s_runner import K8sRunner
            self._runner = K8sRunner(config.runner, workspace, datasets)
        elif config.runner.type == "local":
            from ..runner.local_runner import LocalRunner
            self._runner = LocalRunner(config.runner, workspace, datasets)
        else:
            raise ValueError(f"Unknown runner type: {config.runner.type!r}")

    def run(self, iteration: int, hypothesis_sha: str) -> dict:
        """
        Launch the experiment and return a result dict.

        Raises RunExperimentError on timeout or failure.
        """
        try:
            result = self._runner.run(iteration=iteration, hypothesis_sha=hypothesis_sha)
            return {
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
                "log_tail": result.log_tail,
                "success": True,
            }
        except ExperimentTimeoutError as e:
            raise RunExperimentError(f"Timeout: {e}") from e
        except ExperimentFailedError as e:
            raise RunExperimentError(
                f"Training failed (exit={e.exit_code}): {e}\n\nLog tail:\n{e.log_tail}"
            ) from e
        except Exception as e:
            raise RunExperimentError(f"Unexpected runner error: {e}") from e


class ReadResults(BaseTool):
    name = "read_results"
    description = (
        "Parse the results.json file written by the training run. "
        "Returns the target metric value and all other metrics."
    )

    def __init__(self, config: ConductorConfig):
        self.config = config
        self.results_path = (
            config.session.workspace_path / config.program.results_file
        )
        self.target_metric = config.session.target_metric

    def run(self) -> dict:
        """
        Parse results.json and return metrics.

        Expected results.json format:
        {
            "val_loss": 2.847,
            "val_accuracy": 0.612,
            "epoch": 5,
            "train_loss": 2.103,
            ...
        }

        Raises:
            ResultsNotFoundError if results.json does not exist
            ResultsParseError if the file is not valid JSON or missing target metric
        """
        if not self.results_path.exists():
            raise ResultsNotFoundError(
                f"results.json not found at {self.results_path}. "
                "The training script must write this file before exiting."
            )

        try:
            with open(self.results_path) as f:
                data: dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            raise ResultsParseError(
                f"results.json is not valid JSON: {e}\n"
                f"File contents: {self.results_path.read_text()[:500]}"
            ) from e

        if self.target_metric not in data:
            available = list(data.keys())
            raise ResultsParseError(
                f"Target metric '{self.target_metric}' not found in results.json. "
                f"Available keys: {available}"
            )

        target_value = float(data[self.target_metric])

        return {
            "target_metric": target_value,
            "target_metric_name": self.target_metric,
            "all_metrics": data,
            "epoch": data.get("epoch"),
        }
