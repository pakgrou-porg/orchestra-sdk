"""
orchestra_sdk.runner.local_runner
==================================
LocalRunner — executes train.py directly in the host Python process via
subprocess.  No Docker or GPU required.  Intended for synthetic/CI testing.

The runner:
  1. Spawns `python train.py` in the workspace directory.
  2. Streams stdout/stderr to a buffer (last 100 lines kept as log_tail).
  3. Enforces the configured timeout via subprocess.TimeoutExpired.
  4. Returns a RunResult on success or raises ExperimentFailedError /
     ExperimentTimeoutError on failure.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from .docker_runner import ExperimentFailedError, ExperimentResult, ExperimentTimeoutError
from ..config import RunnerConfig

logger = logging.getLogger(__name__)


class LocalRunner:
    """Run train.py in a subprocess on the local host."""

    def __init__(self, config: RunnerConfig, workspace_dir: Path, datasets_dir: Path) -> None:
        self.config = config
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.datasets_dir = datasets_dir.expanduser().resolve()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, iteration: int, hypothesis_sha: str) -> ExperimentResult:
        """
        Execute train.py in the workspace directory.

        Returns an ExperimentResult on success.
        Raises ExperimentTimeoutError or ExperimentFailedError on failure.
        """
        train_script = self.workspace_dir / "train.py"
        if not train_script.exists():
            raise ExperimentFailedError(
                f"train.py not found in workspace: {self.workspace_dir}",
                exit_code=1,
                log_tail=f"train.py not found in workspace: {self.workspace_dir}",
            )

        env = self._build_env()
        cmd = [sys.executable, str(train_script)]
        timeout = self.config.timeout_seconds

        start = time.monotonic()
        log_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.workspace_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output line by line so the user sees progress
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                logger.info("  [train] %s", line)
                log_lines.append(line)

            proc.wait(timeout=max(1, timeout - int(time.monotonic() - start)))

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise ExperimentTimeoutError(
                f"train.py exceeded timeout of {timeout}s"
            )

        duration = time.monotonic() - start
        log_tail = "\n".join(log_lines[-100:])

        if proc.returncode != 0:
            raise ExperimentFailedError(
                f"train.py exited with code {proc.returncode}",
                exit_code=proc.returncode,
                log_tail=log_tail,
            )

        return ExperimentResult(
            exit_code=0,
            duration_seconds=duration,
            log_tail=log_tail,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_env(self) -> dict[str, str]:
        """Build the subprocess environment."""
        import os
        env = os.environ.copy()
        env["DATASETS_DIR"] = str(self.datasets_dir)
        env["RESULTS_FILE"] = str(self.workspace_dir / "results.json")
        return env
