"""
orchestra_sdk.tools.supabase_tools
=====================================
Supabase persistence tools for the Conductor.
Handles session records, experiment logs, and status updates.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .base import BaseTool, ToolError
from ..config import SupabaseConfig

logger = logging.getLogger(__name__)


class SupabaseToolError(ToolError):
    pass


# ---------------------------------------------------------------------------
# Low-level client wrapper
# ---------------------------------------------------------------------------


class SupabaseClient:
    """
    Thin wrapper around the supabase-py client.
    Provides typed methods for Conductor-specific operations.
    Falls back to local JSONL logging if Supabase is unavailable.
    """

    def __init__(self, config: SupabaseConfig, fallback_dir: Optional[Path] = None):
        self.config = config
        self.fallback_dir = fallback_dir
        self._client = None

    def _get_client(self):
        if self._client is None:
            from supabase import create_client
            self._client = create_client(
                self.config.get_url(),
                self.config.get_key(),
            )
        return self._client

    def _fallback_log(self, table: str, record: dict) -> None:
        """Write to local JSONL if Supabase is unavailable."""
        if not self.fallback_dir:
            return
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        path = self.fallback_dir / f"{table}_fallback.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.warning(f"[SupabaseClient] Wrote fallback record to {path}")

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    def get_session(self, session_name: str) -> Optional[dict]:
        try:
            result = (
                self._get_client()
                .table(self.config.session_table)
                .select("*")
                .eq("name", session_name)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"[SupabaseClient] get_session failed: {e}")
            return None

    def create_session(self, session_name: str, dataset_id: str, config_dict: dict) -> dict:
        record = {
            "name": session_name,
            "dataset_id": dataset_id,
            "status": "running",
            "baseline_metric": None,
            "iteration": 0,
            "config": config_dict,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            result = (
                self._get_client()
                .table(self.config.session_table)
                .insert(record)
                .execute()
            )
            return result.data[0] if result.data else record
        except Exception as e:
            logger.error(f"[SupabaseClient] create_session failed: {e}")
            self._fallback_log(self.config.session_table, record)
            return record

    def update_session(
        self,
        session_name: str,
        baseline_metric: Optional[float] = None,
        status: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> bool:
        updates: dict[str, Any] = {
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        if baseline_metric is not None:
            updates["baseline_metric"] = baseline_metric
        if status is not None:
            updates["status"] = status
        if iteration is not None:
            updates["iteration"] = iteration
        try:
            self._get_client().table(self.config.session_table).update(updates).eq(
                "name", session_name
            ).execute()
            return True
        except Exception as e:
            logger.error(f"[SupabaseClient] update_session failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Experiment operations
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        session_name: str,
        iteration: int,
        hypothesis: str,
        hypothesis_sha: str,
        target_metric: float,
        baseline_metric: float,
        delta: float,
        decision: str,
        duration_seconds: float,
        log_tail: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        record = {
            "session_name": session_name,
            "iteration": iteration,
            "hypothesis": hypothesis,
            "hypothesis_sha": hypothesis_sha,
            "target_metric": target_metric,
            "baseline_at_time": baseline_metric,
            "delta": delta,
            "decision": decision,
            "duration_seconds": duration_seconds,
            "log_tail": log_tail[:2000],  # cap at 2000 chars
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            result = (
                self._get_client()
                .table(self.config.experiments_table)
                .insert(record)
                .execute()
            )
            return result.data[0] if result.data else record
        except Exception as e:
            logger.error(f"[SupabaseClient] log_experiment failed: {e}")
            self._fallback_log(self.config.experiments_table, record)
            return record

    def query_experiments(
        self,
        session_name: str,
        limit: int = 20,
        order_by: str = "iteration",
        decision_filter: Optional[str] = None,
    ) -> list[dict]:
        try:
            query = (
                self._get_client()
                .table(self.config.experiments_table)
                .select("*")
                .eq("session_name", session_name)
                .order(order_by, desc=True)
                .limit(limit)
            )
            if decision_filter:
                query = query.eq("decision", decision_filter)
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"[SupabaseClient] query_experiments failed: {e}")
            return []

    def get_best_metric(self, session_name: str) -> Optional[float]:
        """Return the best (lowest) target metric across all KEEP decisions."""
        experiments = self.query_experiments(
            session_name, limit=500, decision_filter="keep"
        )
        if not experiments:
            return None
        metrics = [e["target_metric"] for e in experiments if e.get("target_metric") is not None]
        return min(metrics) if metrics else None

    def delete_experiments_after_iteration(
        self, session_name: str, max_iteration: int
    ) -> bool:
        """Delete experiment records with iteration > max_iteration to keep Supabase in sync after a reset."""
        try:
            self._get_client().table(self.config.experiments_table).delete().eq(
                "session_name", session_name
            ).gt("iteration", max_iteration).execute()
            return True
        except Exception as e:
            logger.warning(f"[SupabaseClient] delete_experiments_after_iteration failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


class LogToSupabase(BaseTool):
    name = "log_to_supabase"
    description = "Write an experiment record to Supabase"

    def __init__(self, client: SupabaseClient, session_name: str):
        self.client = client
        self.session_name = session_name

    def run(
        self,
        iteration: int,
        hypothesis: str,
        hypothesis_sha: str,
        target_metric: float,
        baseline_metric: float,
        delta: float,
        decision: str,
        duration_seconds: float,
        log_tail: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        return self.client.log_experiment(
            session_name=self.session_name,
            iteration=iteration,
            hypothesis=hypothesis,
            hypothesis_sha=hypothesis_sha,
            target_metric=target_metric,
            baseline_metric=baseline_metric,
            delta=delta,
            decision=decision,
            duration_seconds=duration_seconds,
            log_tail=log_tail,
            metadata=metadata,
        )


class QuerySupabase(BaseTool):
    name = "query_supabase"
    description = "Query experiment history for this session"

    def __init__(self, client: SupabaseClient, session_name: str):
        self.client = client
        self.session_name = session_name

    def run(self, limit: int = 20, order_by: str = "iteration") -> list[dict]:
        return self.client.query_experiments(
            self.session_name, limit=limit, order_by=order_by
        )


class UpdateSession(BaseTool):
    name = "update_session"
    description = "Update the session record (baseline metric, iteration count, status)"

    def __init__(self, client: SupabaseClient, session_name: str):
        self.client = client
        self.session_name = session_name

    def run(
        self,
        baseline_metric: Optional[float] = None,
        status: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> dict:
        success = self.client.update_session(
            self.session_name,
            baseline_metric=baseline_metric,
            status=status,
            iteration=iteration,
        )
        return {"updated": success}
