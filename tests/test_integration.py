"""
Integration test: runs a 3-iteration loop with mocked LLM and runner.
Validates the full keep/discard cycle end-to-end.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestra_sdk.config import ConductorConfig
from orchestra_sdk.context import Hypothesis, HypothesisEdit
from orchestra_sdk.loop import ConductorLoop, Decision


@pytest.fixture
def config(workspace: Path, minimal_config_dict: dict, set_test_env) -> ConductorConfig:
    return ConductorConfig.model_validate(minimal_config_dict)


def make_hypothesis(find: str, replace: str, confidence: float = 0.8) -> Hypothesis:
    return Hypothesis(
        hypothesis=f"Change {find!r} to {replace!r}",
        change_type="hyperparameter",
        expected_effect="reduce val_loss",
        risk="low",
        confidence=confidence,
        memory_note="Tested a hyperparameter change.",
        edit=HypothesisEdit(find=find, replace=replace),
    )


class TestConductorLoopIntegration:
    """
    Full loop integration test with mocked LLM and runner.
    Tests: initialization, hypothesis application, keep/discard logic, Supabase logging.
    """

    @pytest.mark.asyncio
    async def test_three_iteration_loop(self, config: ConductorConfig, workspace: Path):
        """
        Iteration 1: metric improves (KEEP)
        Iteration 2: metric worsens (DISCARD)
        Iteration 3: metric improves again (KEEP)
        """
        # Mock LLM to return valid hypotheses (max_iterations=5, need 5)
        hypotheses = [
            make_hypothesis("LEARNING_RATE = 0.01", "LEARNING_RATE = 0.001"),
            make_hypothesis("BATCH_SIZE = 32", "BATCH_SIZE = 64"),
            make_hypothesis("DROPOUT = 0.1", "DROPOUT = 0.2"),
            make_hypothesis("EPOCHS = 5", "EPOCHS = 3"),
            make_hypothesis("DROPOUT = 0.2", "DROPOUT = 0.05"),
        ]
        hypothesis_iter = iter(hypotheses)

        # Metrics: iter1=2.800 (KEEP, baseline), iter2=2.900 (DISCARD), iter3=2.750 (KEEP),
        #          iter4=2.800 (DISCARD), iter5=2.700 (KEEP)
        metrics = [2.800, 2.900, 2.750, 2.800, 2.700]
        metric_iter = iter(metrics)

        loop = ConductorLoop(config)

        with (
            patch.object(loop.llm, "structured_output", new_callable=AsyncMock) as mock_llm,
            patch.object(loop.run_experiment, "run") as mock_runner,
            patch.object(loop.supabase, "log_experiment") as mock_log,
            patch.object(loop.supabase, "create_session") as mock_create,
            patch.object(loop.supabase, "get_session", return_value=None),
            patch.object(loop.supabase, "update_session"),
            patch.object(loop.supabase, "query_experiments", return_value=[]),
        ):
            mock_create.return_value = {"id": "test-session-id"}
            async def _next_hyp(*a, **kw):
                return next(hypothesis_iter)
            mock_llm.side_effect = _next_hyp

            def mock_run(**kwargs):
                metric = next(metric_iter)
                # Write results.json
                (workspace / "results.json").write_text(
                    json.dumps({"val_loss": metric, "epoch": 5})
                )
                return {"exit_code": 0, "duration_seconds": 10.0, "log_tail": "done", "success": True}

            mock_runner.side_effect = mock_run

            await loop.run()

        # Verify all 5 iterations ran
        assert loop.iteration == 5

        # Verify keep/discard logic: best metric is 2.700 (iter5)
        assert loop.baseline_metric == pytest.approx(2.700)

        # Verify Supabase logging was called 5 times
        assert mock_log.call_count == 5

    @pytest.mark.asyncio
    async def test_failed_experiment_reverts(self, config: ConductorConfig, workspace: Path):
        """When an experiment fails, the workspace should revert to last keep SHA."""
        from orchestra_sdk.tools.run_experiment import RunExperimentError

        loop = ConductorLoop(config)

        with (
            patch.object(loop.llm, "structured_output", new_callable=AsyncMock) as mock_llm,
            patch.object(loop.run_experiment, "run") as mock_runner,
            patch.object(loop.supabase, "log_experiment"),
            patch.object(loop.supabase, "create_session", return_value={"id": "x"}),
            patch.object(loop.supabase, "get_session", return_value=None),
            patch.object(loop.supabase, "update_session"),
            patch.object(loop.supabase, "query_experiments", return_value=[]),
        ):
            mock_llm.return_value = make_hypothesis("LEARNING_RATE = 0.01", "LEARNING_RATE = 0.0001")
            mock_runner.side_effect = RunExperimentError("Container OOM")

            result = await loop._run_iteration(1)

        assert result.decision == Decision.FAILED
        assert "OOM" in result.error

    @pytest.mark.asyncio
    async def test_skips_when_train_py_missing(self, config: ConductorConfig, workspace: Path):
        """When train.py is missing, the iteration should be skipped gracefully."""
        (workspace / "train.py").unlink()

        loop = ConductorLoop(config)

        with (
            patch.object(loop.supabase, "create_session", return_value={"id": "x"}),
            patch.object(loop.supabase, "get_session", return_value=None),
            patch.object(loop.supabase, "update_session"),
            patch.object(loop.supabase, "query_experiments", return_value=[]),
            patch.object(loop.supabase, "log_experiment"),
        ):
            result = await loop._run_iteration(1)

        assert result.decision == Decision.SKIPPED
        assert "train.py" in result.error
