"""
orchestra_sdk.runner.docker_runner
====================================
Docker-based experiment runner.
Launches training jobs as ephemeral containers with GPU access.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import RunnerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExperimentTimeoutError(Exception):
    pass


class ExperimentFailedError(Exception):
    def __init__(self, message: str, exit_code: int, log_tail: str):
        super().__init__(message)
        self.exit_code = exit_code
        self.log_tail = log_tail


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    exit_code: int
    duration_seconds: float
    log_tail: str  # last 50 lines
    container_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Docker runner
# ---------------------------------------------------------------------------


class DockerRunner:
    """
    Runs a training job inside a Docker container.

    The container receives:
      - /workspace (rw) — the session workspace (train.py, results.json, etc.)
      - /datasets  (ro) — the datasets directory
      - ITERATION env var — current iteration number
      - ORCHESTRA_SESSION env var — session name
    """

    def __init__(self, config: RunnerConfig, workspace_dir: Path, datasets_dir: Path):
        self.config = config
        self.workspace_dir = workspace_dir
        self.datasets_dir = datasets_dir
        self._client = None

    def _get_client(self):
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    def _build_gpu_config(self):
        """Build Docker device_requests for GPU access."""
        import docker
        device = self.config.gpu_device
        if device == "none":
            return []
        if device == "all":
            return [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        # Specific device index
        return [docker.types.DeviceRequest(device_ids=[device], capabilities=[["gpu"]])]

    def run(self, iteration: int, hypothesis_sha: str) -> ExperimentResult:
        """
        Launch the training container and wait for completion.

        Raises:
            ExperimentTimeoutError if the container exceeds config.timeout_seconds
            ExperimentFailedError if the container exits with non-zero code
        """
        client = self._get_client()
        start_time = time.time()

        container_kwargs = {
            "image": self.config.image,
            "command": "python /workspace/train.py",
            "volumes": {
                str(self.workspace_dir): {"bind": "/workspace", "mode": "rw"},
                str(self.datasets_dir): {"bind": "/datasets", "mode": "ro"},
            },
            "environment": {
                "ITERATION": str(iteration),
                "ORCHESTRA_SESSION": hypothesis_sha[:8],
                "HYPOTHESIS_SHA": hypothesis_sha,
            },
            "remove": False,  # We need to read logs before removing
            "detach": True,
        }

        # Add GPU config if not "none"
        if self.config.gpu_device != "none":
            container_kwargs["device_requests"] = self._build_gpu_config()

        logger.info(
            f"[DockerRunner] Starting container: {self.config.image} "
            f"(iteration={iteration}, sha={hypothesis_sha[:8]})"
        )

        try:
            container = client.containers.run(**container_kwargs)
        except Exception as e:
            raise ExperimentFailedError(
                f"Failed to start container: {e}", exit_code=-1, log_tail=""
            )

        # Poll until done or timeout
        try:
            result = container.wait(timeout=self.config.timeout_seconds)
        except Exception:
            container.kill()
            container.remove(force=True)
            duration = time.time() - start_time
            raise ExperimentTimeoutError(
                f"Container exceeded timeout of {self.config.timeout_seconds}s "
                f"after {duration:.1f}s"
            )

        duration = time.time() - start_time
        exit_code = result.get("StatusCode", -1)

        # Collect logs (last 50 lines)
        try:
            logs = container.logs(tail=50).decode("utf-8", errors="replace")
        except Exception:
            logs = "(log collection failed)"

        container.remove(force=True)

        if exit_code != 0:
            raise ExperimentFailedError(
                f"Container exited with code {exit_code}",
                exit_code=exit_code,
                log_tail=logs,
            )

        return ExperimentResult(
            exit_code=exit_code,
            duration_seconds=duration,
            log_tail=logs,
        )

    def is_available(self) -> bool:
        """Check if Docker daemon is reachable."""
        try:
            self._get_client().ping()
            return True
        except Exception:
            return False
