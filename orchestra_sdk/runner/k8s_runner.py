"""
orchestra_sdk.runner.k8s_runner
=================================
Kubernetes Job-based experiment runner.
Creates a K8s Job, polls until completion, and cleans up.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import RunnerConfig
from .docker_runner import ExperimentFailedError, ExperimentTimeoutError

logger = logging.getLogger(__name__)


@dataclass
class K8sExperimentResult:
    exit_code: int
    duration_seconds: float
    log_tail: str
    job_name: str


class K8sRunner:
    """
    Runs a training job as a Kubernetes batch/v1 Job.

    Requires:
      - A PersistentVolumeClaim named 'orchestra-workspace' in the target namespace
      - A PersistentVolumeClaim named 'orchestra-datasets' (ReadOnlyMany)
      - NVIDIA device plugin or ROCm device plugin for GPU access
    """

    def __init__(self, config: RunnerConfig, workspace_dir: Path, datasets_dir: Path):
        self.config = config
        self.workspace_dir = workspace_dir
        self.datasets_dir = datasets_dir
        self._batch_v1 = None
        self._core_v1 = None

    def _get_clients(self):
        if self._batch_v1 is None:
            from kubernetes import client as k8s_client, config as k8s_config
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()
            self._batch_v1 = k8s_client.BatchV1Api()
            self._core_v1 = k8s_client.CoreV1Api()
        return self._batch_v1, self._core_v1

    def _build_job_manifest(self, job_name: str, iteration: int, hypothesis_sha: str) -> dict:
        """Build the K8s Job manifest."""
        resources = {
            "requests": {"memory": "4Gi", "cpu": "2"},
            "limits": {"memory": "16Gi", "cpu": "8"},
        }
        if self.config.gpu_device != "none":
            resources["limits"]["nvidia.com/gpu"] = "1"

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": "orchestra-musician",
                    "session": hypothesis_sha[:8],
                    "iteration": str(iteration),
                },
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": self.config.k8s_job_ttl_seconds,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "nodeSelector": self.config.node_selector or {},
                        "containers": [
                            {
                                "name": "musician",
                                "image": self.config.image,
                                "command": ["python", "/workspace/train.py"],
                                "env": [
                                    {"name": "ITERATION", "value": str(iteration)},
                                    {"name": "HYPOTHESIS_SHA", "value": hypothesis_sha},
                                    {"name": "ORCHESTRA_SESSION", "value": hypothesis_sha[:8]},
                                ],
                                "resources": resources,
                                "volumeMounts": [
                                    {"name": "workspace", "mountPath": "/workspace"},
                                    {"name": "datasets", "mountPath": "/datasets", "readOnly": True},
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "workspace",
                                "persistentVolumeClaim": {"claimName": "orchestra-workspace"},
                            },
                            {
                                "name": "datasets",
                                "persistentVolumeClaim": {
                                    "claimName": "orchestra-datasets",
                                    "readOnly": True,
                                },
                            },
                        ],
                    }
                },
            },
        }

    def run(self, iteration: int, hypothesis_sha: str) -> K8sExperimentResult:
        """Launch a K8s Job and wait for completion."""
        batch_v1, core_v1 = self._get_clients()
        from kubernetes import client as k8s_client

        job_name = f"orchestra-musician-{iteration}-{uuid.uuid4().hex[:6]}"
        manifest = self._build_job_manifest(job_name, iteration, hypothesis_sha)

        start_time = time.time()
        logger.info(f"[K8sRunner] Creating job: {job_name}")

        batch_v1.create_namespaced_job(
            namespace=self.config.namespace,
            body=manifest,
        )

        # Poll until complete or timeout.
        # Transient API errors (network blips, apiserver restarts) are retried
        # up to _MAX_POLL_ERRORS consecutive times before the iteration is failed.
        _MAX_POLL_ERRORS = 3
        _consecutive_errors = 0
        poll_interval = 10
        elapsed = 0
        while elapsed < self.config.timeout_seconds:
            time.sleep(poll_interval)
            elapsed = time.time() - start_time

            try:
                job = batch_v1.read_namespaced_job(
                    name=job_name, namespace=self.config.namespace
                )
                _consecutive_errors = 0  # reset on success
            except Exception as poll_err:
                _consecutive_errors += 1
                logger.warning(
                    f"[K8sRunner] read_namespaced_job transient error "
                    f"({_consecutive_errors}/{_MAX_POLL_ERRORS}): {poll_err}"
                )
                if _consecutive_errors >= _MAX_POLL_ERRORS:
                    raise ExperimentFailedError(
                        f"K8sAPI unreachable after {_MAX_POLL_ERRORS} consecutive "
                        f"poll errors for job {job_name}: {poll_err}",
                        exit_code=-1,
                        log_tail="",
                    ) from poll_err
                continue

            status = job.status

            if status.succeeded and status.succeeded > 0:
                # Job completed successfully
                log_tail = self._get_pod_logs(core_v1, job_name)
                return K8sExperimentResult(
                    exit_code=0,
                    duration_seconds=elapsed,
                    log_tail=log_tail,
                    job_name=job_name,
                )

            if status.failed and status.failed > 0:
                log_tail = self._get_pod_logs(core_v1, job_name)
                raise ExperimentFailedError(
                    f"K8s Job {job_name} failed after {elapsed:.1f}s",
                    exit_code=-1,
                    log_tail=log_tail,
                )

        # Timeout — delete the job
        try:
            batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.config.namespace,
                body=k8s_client.V1DeleteOptions(propagation_policy="Foreground"),
            )
        except Exception:
            pass
        raise ExperimentTimeoutError(
            f"K8s Job {job_name} exceeded timeout of {self.config.timeout_seconds}s"
        )

    def _get_pod_logs(self, core_v1, job_name: str) -> str:
        """Retrieve logs from the pod created by the job."""
        try:
            pods = core_v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector=f"job-name={job_name}",
            )
            if pods.items:
                pod_name = pods.items[0].metadata.name
                logs = core_v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.config.namespace,
                    tail_lines=50,
                )
                return logs
        except Exception as e:
            return f"(log collection failed: {e})"
        return "(no pods found)"
