"""
orchestra_sdk.loop
===================
The ConductorLoop: autonomous LLM-driven research loop.
Implements the 10-step per-iteration cycle with keep/discard logic.
"""

from __future__ import annotations

import asyncio
import datetime
import json as _json
import logging
import shutil
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import ConductorConfig
from .context import Hypothesis, assemble_context
from .llm import LLMClient, LLMError, Message, StructuredOutputError
from .memory.store import MemoryStore
from .tools.file_tools import EditFile, EditFileError, ReadFile
from .tools.git_tools import GitCommitError, GitManager
from .tools.memory_tools import AddMemory, SearchMemories
from .tools.run_experiment import ReadResults, ResultsNotFoundError, ResultsParseError, RunExperiment, RunExperimentError
from .tools.supabase_tools import SupabaseClient

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    KEEP = "keep"
    DISCARD = "discard"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Iteration result
# ---------------------------------------------------------------------------


@dataclass
class IterationResult:
    iteration: int
    decision: Decision
    hypothesis: Optional[str] = None
    target_metric: Optional[float] = None
    baseline_metric: Optional[float] = None
    delta: Optional[float] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    hypothesis_sha: Optional[str] = None


# ---------------------------------------------------------------------------
# ConductorLoop
# ---------------------------------------------------------------------------


class ConductorLoop:
    """
    Autonomous research loop. Runs up to config.session.max_iterations iterations.

    Each iteration:
      1. Read program.md
      2. Search memories (semantic)
      3. Read git log
      4. Read metric history from Supabase
      5. Propose hypothesis (LLM structured output)
      6. Apply edit to train.py
      7. Commit candidate
      8. Run experiment (Docker/K8s)
      9. Read results
      10. Keep or discard
      11. Log to Supabase
      12. Add memory
    """

    def __init__(self, config: ConductorConfig, resume: bool = False):
        self.config = config
        self.resume = resume

        # Core components
        self.llm = LLMClient(config.llm)
        self.git = GitManager(config.session.workspace_path, config.session.branch)
        self.supabase = SupabaseClient(
            config.supabase,
            fallback_dir=config.session.workspace_path / ".orchestra_fallback",
        )
        self.memory_store = MemoryStore(
            config.supabase, config.memory, config.session.name
        )

        # Tools
        workspace = config.session.workspace_path
        self.read_file = ReadFile(workspace)
        self.edit_file = EditFile(workspace)
        self.run_experiment = RunExperiment(config)
        self.read_results = ReadResults(config)
        self.search_memories = SearchMemories(self.memory_store)
        self.add_memory = AddMemory(self.memory_store)

        # State
        # NOTE: baseline_metric is the *rolling last-kept* metric, not a fixed initial
        # measurement. It advances on every KEEP. The name is retained for Supabase
        # column compatibility; treat it as "last_kept_metric" in logic.
        self.baseline_metric: Optional[float] = None
        self.best_metric: Optional[float] = None       # best score ever achieved this session
        self.best_model_path: Optional[Path] = None    # path to the preserved best model
        self.iteration: int = 0
        self.last_keep_sha: Optional[str] = None
        self._session_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """
        Install SIGTERM and SIGINT handlers so that process-manager evictions
        (systemd, OKE pod eviction) result in a clean shutdown rather than
        leaving the session in 'running' status with a dirty workspace.
        The handler sets a flag; the main loop checks it after each iteration
        so we never interrupt mid-step.
        """
        self._shutdown_requested = False

        def _handle_signal(signum, frame):  # noqa: ARG001
            sig_name = signal.Signals(signum).name
            console.print(
                f"\n[yellow]Received {sig_name} — finishing current step then shutting down…[/yellow]"
            )
            self._shutdown_requested = True

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    async def _graceful_shutdown(self, results: list["IterationResult"]) -> None:
        """Revert workspace to last KEEP and mark session stopped in Supabase."""
        console.print("[yellow]Graceful shutdown: reverting workspace to last KEEP…[/yellow]")
        if self.last_keep_sha:
            try:
                self.git.reset.run(self.last_keep_sha)
                console.print(
                    f"[green]Workspace reverted to {self.last_keep_sha[:8]}[/green]"
                )
            except Exception as e:
                logger.warning(f"Workspace revert failed during shutdown: {e}")
        self.supabase.update_session(
            self.config.session.name,
            status="stopped",
            iteration=self.iteration,
        )
        self._print_final_summary(results)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def _initialize(self) -> None:
        """Set up workspace, git repo, and session record."""
        console.print(f"\n[cyan]Initializing session:[/cyan] {self.config.session.name}")

        # Ensure workspace exists and git is initialized
        self.git.initialize()

        # Validate eval script exists
        eval_path = self.config.session.workspace_path / self.config.program.eval_script
        if not eval_path.exists():
            console.print(
                f"[yellow]Warning:[/yellow] eval_script not found at {eval_path}. "
                "Create it before running experiments."
            )
        self._eval_path = eval_path if eval_path.exists() else None

        # Create or resume session in Supabase
        existing = self.supabase.get_session(self.config.session.name)
        if existing and self.resume:
            self.baseline_metric = existing.get("baseline_metric")
            self.iteration = existing.get("iteration", 0)
            self._session_id = existing.get("id")
            console.print(
                f"[green]Resuming session[/green] at iteration {self.iteration}, "
                f"baseline={self.baseline_metric}"
            )
        else:
            session = self.supabase.create_session(
                session_name=self.config.session.name,
                dataset_id=self.config.session.dataset_id,
                config_dict=self.config.model_dump(),
            )
            self._session_id = session.get("id")
            console.print(f"[green]Session created[/green] (id={self._session_id})")

        # Commit initial workspace state so git reset never removes program.md or train.py
        try:
            initial_sha = self.git.current_sha()
            if initial_sha is None:
                # Nothing committed yet — commit the initial workspace
                self.git.commit.run("[INIT] Initial workspace state")
        except Exception:
            # If git isn't initialized yet, initialize() already ran git init
            try:
                self.git.commit.run("[INIT] Initial workspace state")
            except Exception:
                pass

        # Set last keep SHA to current HEAD (after initial commit)
        self.last_keep_sha = self.git.current_sha()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the autonomous loop until max_iterations or target_value reached."""
        await self._initialize()
        self._install_signal_handlers()

        console.print(
            Panel(
                f"Starting loop: max {self.config.session.max_iterations} iterations\n"
                f"Target: {self.config.session.target_metric} ≤ "
                f"{self.config.session.target_value or 'N/A'}",
                title="[cyan]CONDUCTOR LOOP[/cyan]",
                border_style="cyan",
            )
        )

        results: list[IterationResult] = []

        while self.iteration < self.config.session.max_iterations:
            self.iteration += 1
            console.rule(f"[cyan]Iteration {self.iteration}[/cyan]")

            result = await self._run_iteration(self.iteration)
            results.append(result)

            self._print_iteration_summary(result)

            # Check graceful shutdown requested by SIGTERM/SIGINT
            if self._shutdown_requested:
                await self._graceful_shutdown(results)
                return

            # Check target reached
            if (
                self.config.session.target_value is not None
                and result.target_metric is not None
                and result.target_metric <= self.config.session.target_value
            ):
                console.print(
                    f"\n[green bold]TARGET REACHED![/green bold] "
                    f"{self.config.session.target_metric}={result.target_metric:.4f} "
                    f"≤ {self.config.session.target_value}"
                )
                break

        # Mark session complete
        self.supabase.update_session(
            self.config.session.name,
            status="completed",
            iteration=self.iteration,
        )
        self._print_final_summary(results)

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    async def _run_iteration(self, iteration: int) -> IterationResult:
        start_time = time.time()

        # --- Step 1: Read program ---
        try:
            program_text = self.read_file.run(self.config.program.path)
        except FileNotFoundError:
            return IterationResult(
                iteration=iteration,
                decision=Decision.SKIPPED,
                error=f"program.md not found at {self.config.program.path}",
            )

        # --- Step 2: Search memories ---
        memories = []
        if self.config.memory.enabled:
            try:
                memories = self.memory_store.search(
                    f"iteration {iteration} {self.config.session.target_metric}"
                )
            except Exception as e:
                logger.warning(f"Memory search failed: {e}")

        # --- Step 3: Read git log ---
        try:
            git_log = self.git.log.run(n=10)
        except Exception as e:
            logger.warning(f"Git log failed: {e}")
            git_log = []

        # --- Step 4: Read metric history ---
        metric_history = self.supabase.query_experiments(
            self.config.session.name, limit=20
        )

        # --- Step 5: Read current train.py ---
        try:
            train_script = self.read_file.run(self.config.program.train_script)
        except FileNotFoundError:
            return IterationResult(
                iteration=iteration,
                decision=Decision.SKIPPED,
                error=f"train.py not found at {self.config.program.train_script}",
            )

        # --- Step 6: Propose hypothesis ---
        console.print(f"  [dim]→ Proposing hypothesis...[/dim]")
        messages = assemble_context(
            config=self.config,
            program_text=program_text,
            train_script_text=train_script,
            git_log=git_log,
            metric_history=metric_history,
            memories=memories,
            baseline_metric=self.baseline_metric,
            iteration=iteration,
        )

        try:
            hypothesis: Hypothesis = await self.llm.structured_output(
                messages, Hypothesis, max_retries=2
            )
        except StructuredOutputError as e:
            return IterationResult(
                iteration=iteration,
                decision=Decision.SKIPPED,
                error=f"LLM failed to produce valid hypothesis: {e}",
            )
        except LLMError as e:
            return IterationResult(
                iteration=iteration,
                decision=Decision.SKIPPED,
                error=f"LLM error: {e}",
            )

        console.print(
            f"  [cyan]Hypothesis:[/cyan] {hypothesis.hypothesis}\n"
            f"  [dim]Type: {hypothesis.change_type} | Risk: {hypothesis.risk} | "
            f"Confidence: {hypothesis.confidence:.0%}[/dim]"
        )

        # --- Step 7: Apply edit ---
        console.print(f"  [dim]→ Applying edit to {self.config.program.train_script}...[/dim]")
        try:
            edit_result = self.edit_file.run(
                path=self.config.program.train_script,
                find=hypothesis.edit.find,
                replace=hypothesis.edit.replace,
                validate_single_match=True,
            )
        except EditFileError as e:
            return IterationResult(
                iteration=iteration,
                decision=Decision.SKIPPED,
                hypothesis=hypothesis.hypothesis,
                error=f"Edit failed: {e}",
            )

        # --- Step 8: Commit candidate ---
        commit_message = (
            f"[CANDIDATE] iter {iteration}: {hypothesis.hypothesis}\n\n"
            f"change_type: {hypothesis.change_type}\n"
            f"risk: {hypothesis.risk}\n"
            f"confidence: {hypothesis.confidence}"
        )
        try:
            commit_info = self.git.commit.run(commit_message)
            hypothesis_sha = commit_info["sha"]
        except GitCommitError as e:
            return IterationResult(
                iteration=iteration,
                decision=Decision.SKIPPED,
                hypothesis=hypothesis.hypothesis,
                error=f"Git commit failed: {e}",
            )

        console.print(f"  [dim]→ Committed: {hypothesis_sha[:8]}[/dim]")

        # --- Step 9: Run experiment ---
        console.print(f"  [dim]→ Running experiment...[/dim]")
        experiment_log = ""
        try:
            exp_result = self.run_experiment.run(
                iteration=iteration,
                hypothesis_sha=hypothesis_sha,
            )
            experiment_log = exp_result.get("log_tail", "")
        except RunExperimentError as e:
            # Revert to last keep
            if self.last_keep_sha:
                self.git.reset.run(self.last_keep_sha)
            duration = time.time() - start_time
            result = IterationResult(
                iteration=iteration,
                decision=Decision.FAILED,
                hypothesis=hypothesis.hypothesis,
                hypothesis_sha=hypothesis_sha,
                baseline_metric=self.baseline_metric,
                duration_seconds=duration,
                error=str(e),
            )
            await self._log_and_memorize(result, hypothesis, str(e))
            return result

        # --- Step 10: Read results.json (written by train.py) ---
        try:
            results_data = self.read_results.run()
            target_metric = results_data["target_metric"]
        except (ResultsNotFoundError, ResultsParseError) as e:
            if self.last_keep_sha:
                self.git.reset.run(self.last_keep_sha)
            duration = time.time() - start_time
            result = IterationResult(
                iteration=iteration,
                decision=Decision.FAILED,
                hypothesis=hypothesis.hypothesis,
                hypothesis_sha=hypothesis_sha,
                baseline_metric=self.baseline_metric,
                duration_seconds=duration,
                error=str(e),
            )
            await self._log_and_memorize(result, hypothesis, str(e))
            return result

        # --- Step 10b: Run evaluate.py if present ---
        if self._eval_path is not None:
            console.print(f"  [dim]→ Running evaluate.py...[/dim]")
            try:
                import subprocess
                eval_proc = subprocess.run(
                    ["python", str(self._eval_path)],
                    cwd=str(self.config.session.workspace_path),
                    capture_output=True,
                    text=True,
                    timeout=self.config.runner.timeout_seconds,
                )
                if eval_proc.returncode != 0:
                    logger.warning(
                        f"evaluate.py exited {eval_proc.returncode}: "
                        f"{eval_proc.stderr[:300]}"
                    )
                else:
                    logger.debug(f"evaluate.py stdout: {eval_proc.stdout[:300]}")
            except subprocess.TimeoutExpired:
                logger.warning("evaluate.py timed out; continuing without eval output")
            except Exception as e:
                logger.warning(f"evaluate.py failed to run: {e}")

        # --- Step 11: Keep or discard ---
        duration = time.time() - start_time

        # First iteration: always KEEP to establish baseline
        if self.baseline_metric is None:
            delta = 0.0
            decision = Decision.KEEP
            self.baseline_metric = target_metric
            self.last_keep_sha = hypothesis_sha
            keep_message = (
                f"[KEEP] iter {iteration} (baseline): {hypothesis.hypothesis}\n"
                f"metric={target_metric:.4f} (first iteration — baseline established)"
            )
            try:
                self.git.commit.run(keep_message)
            except GitCommitError:
                pass
            result = IterationResult(
                iteration=iteration,
                decision=decision,
                hypothesis=hypothesis.hypothesis,
                hypothesis_sha=hypothesis_sha,
                target_metric=target_metric,
                baseline_metric=target_metric,
                delta=delta,
                duration_seconds=duration,
            )
            await self._log_and_memorize(result, hypothesis, experiment_log)
            self.supabase.update_session(
                self.config.session.name,
                baseline_metric=self.baseline_metric,
                iteration=iteration,
            )
            return result

        baseline = self.baseline_metric
        delta = target_metric - baseline

        if delta <= self.config.session.keep_threshold:
            decision = Decision.KEEP
            # Update baseline and last keep SHA
            self.baseline_metric = target_metric
            self.last_keep_sha = hypothesis_sha
            # Tag the commit as KEEP
            keep_message = (
                f"[KEEP] iter {iteration}: {hypothesis.hypothesis}\n"
                f"metric={target_metric:.4f} delta={delta:+.4f}"
            )
            try:
                self.git.commit.run(keep_message)
            except GitCommitError:
                pass  # Not critical

            # Preserve best model: if this is better than all-time best, copy to best/ dir
            if self.best_metric is None or target_metric < self.best_metric:
                self.best_metric = target_metric
                self.best_model_path = await self._save_best_model(iteration, target_metric)
                if self.best_model_path:
                    console.print(
                        f"  [bold green]NEW BEST[/bold green] metric={target_metric:.4f} "
                        f"→ saved to {self.best_model_path}"
                    )
        else:
            decision = Decision.DISCARD
            # Revert to last keep
            if self.last_keep_sha:
                self.git.reset.run(self.last_keep_sha)

        result = IterationResult(
            iteration=iteration,
            decision=decision,
            hypothesis=hypothesis.hypothesis,
            hypothesis_sha=hypothesis_sha,
            target_metric=target_metric,
            baseline_metric=baseline,
            delta=delta,
            duration_seconds=duration,
        )

        await self._log_and_memorize(result, hypothesis, experiment_log)

        # Update session in Supabase
        self.supabase.update_session(
            self.config.session.name,
            baseline_metric=self.baseline_metric,
            iteration=iteration,
        )

        return result

    # ------------------------------------------------------------------
    # Best model preservation
    # ------------------------------------------------------------------

    async def _save_best_model(self, iteration: int, metric: float) -> Optional[Path]:
        """
        Copy the trained model output to a dedicated 'best/' directory.

        The Musician container is expected to write its model artifacts to
        `workspace/output/` (or the path in config.program.results_file's parent).
        We copy that directory to `workspace/best/` and write a manifest.

        Returns the path to the best/ directory, or None if nothing to copy.
        """

        workspace = self.config.session.workspace_path
        output_dir = workspace / "output"
        best_dir = workspace / "best"

        if not output_dir.exists():
            # Fallback: copy the whole workspace snapshot via git archive
            logger.debug("output/ dir not found; skipping model copy")
            return None

        try:
            # Remove previous best and replace with current output
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(output_dir, best_dir)

            # Write a manifest so we know which iteration produced this best
            manifest = {
                "session": self.config.session.name,
                "iteration": iteration,
                "metric": metric,
                "git_sha": self.last_keep_sha,
                "timestamp": datetime.datetime.utcnow().isoformat() + 'Z',
            }
            (best_dir / "best_manifest.json").write_text(
                _json.dumps(manifest, indent=2)
            )

            # Also update Supabase session_best_runs table if it exists
            try:
                self.supabase._get_client().table("session_best_runs").upsert({
                    "session_name": self.config.session.name,
                    "iteration": iteration,
                    "metric": metric,
                    "git_sha": self.last_keep_sha or "",
                    "model_path": str(best_dir),
                }, on_conflict="session_name").execute()
            except Exception as e:
                logger.debug(f"session_best_runs upsert skipped: {e}")

            return best_dir
        except Exception as e:
            logger.warning(f"Failed to save best model: {e}")
            return None

    # ------------------------------------------------------------------
    # Logging and memory
    # ------------------------------------------------------------------

    async def _log_and_memorize(
        self,
        result: IterationResult,
        hypothesis: Optional[Hypothesis],
        log_tail: str,
    ) -> None:
        """Log to Supabase and add a memory."""
        self.supabase.log_experiment(
            session_name=self.config.session.name,
            iteration=result.iteration,
            hypothesis=result.hypothesis or "(none)",
            hypothesis_sha=result.hypothesis_sha or "",
            target_metric=result.target_metric or 0.0,
            baseline_metric=result.baseline_metric or 0.0,
            delta=result.delta or 0.0,
            decision=result.decision.value,
            duration_seconds=result.duration_seconds,
            log_tail=log_tail,
            metadata={"error": result.error} if result.error else {},
        )

        if hypothesis and self.config.memory.enabled:
            memory_content = (
                f"{hypothesis.memory_note} "
                f"Result: {result.decision.value}. "
                f"Metric: {result.target_metric:.4f if result.target_metric else 'N/A'}. "
                f"Delta: {result.delta:+.4f if result.delta else 'N/A'}."
            )
            self.add_memory.run(
                content=memory_content,
                iteration=result.iteration,
                decision=result.decision.value,
            )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _print_iteration_summary(self, result: IterationResult) -> None:
        color = {
            Decision.KEEP: "green",
            Decision.DISCARD: "yellow",
            Decision.FAILED: "red",
            Decision.SKIPPED: "dim",
        }.get(result.decision, "white")

        metric_str = f"{result.target_metric:.4f}" if result.target_metric is not None else "N/A"
        delta_str = f"{result.delta:+.4f}" if result.delta is not None else "N/A"

        console.print(
            f"  [{color}]{result.decision.value.upper()}[/{color}] "
            f"metric={metric_str} Δ={delta_str} "
            f"({result.duration_seconds:.1f}s)"
        )
        if result.error:
            console.print(f"  [red]Error:[/red] {result.error[:200]}")

    def _print_final_summary(self, results: list[IterationResult]) -> None:
        keeps = [r for r in results if r.decision == Decision.KEEP]
        discards = [r for r in results if r.decision == Decision.DISCARD]
        failures = [r for r in results if r.decision == Decision.FAILED]
        skips = [r for r in results if r.decision == Decision.SKIPPED]

        table = Table(title="Session Summary", border_style="cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        table.add_row("Total iterations", str(len(results)))
        table.add_row("KEEP", f"[green]{len(keeps)}[/green]")
        table.add_row("DISCARD", f"[yellow]{len(discards)}[/yellow]")
        table.add_row("FAILED", f"[red]{len(failures)}[/red]")
        table.add_row("SKIPPED", f"[dim]{len(skips)}[/dim]")
        if self.baseline_metric is not None:
            table.add_row("Final baseline", f"{self.baseline_metric:.4f}")
        if self.best_metric is not None:
            table.add_row("Best metric", f"[bold green]{self.best_metric:.4f}[/bold green]")
        if self.best_model_path is not None:
            table.add_row("Best model saved", f"[cyan]{self.best_model_path}[/cyan]")
        console.print(table)
