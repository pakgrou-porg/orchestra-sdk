"""
orchestra_sdk.__main__
======================
CLI entry point. Run with:
    python -m orchestra_sdk.conductor --help
    orchestra --help  (after pip install)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="orchestra",
    help="Orchestra Conductor — autonomous LLM research loop for LLM fine-tuning",
    add_completion=False,
)
console = Console()


def _load_env(env_file: Optional[Path]) -> None:
    """Load .env file if provided or if .env exists in cwd."""
    if env_file and env_file.exists():
        load_dotenv(env_file)
    elif Path(".env").exists():
        load_dotenv(".env")


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to conductor_config.yaml", exists=True
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file with API keys"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate config and environment, then exit"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", "-n", help="Override max_iterations from config"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume an existing session from Supabase"
    ),
) -> None:
    """Run the Conductor autonomous loop for a session."""
    _load_env(env_file)

    from orchestra_sdk.config import ConductorConfig

    try:
        cfg = ConductorConfig.from_yaml(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(1)

    if max_iterations is not None:
        cfg.session.max_iterations = max_iterations

    # Print summary
    console.print(
        Panel(
            cfg.summary(),
            title="[cyan]ORCHESTRA CONDUCTOR[/cyan]",
            border_style="cyan",
        )
    )

    # Validate environment
    errors = cfg.validate_environment()
    if errors:
        console.print("\n[red]Environment errors:[/red]")
        for err in errors:
            console.print(f"  [red]✗[/red] {err}")
        if not dry_run:
            raise typer.Exit(1)
    else:
        console.print("\n[green]✓ Environment validated[/green]")

    if dry_run:
        console.print("\n[yellow]--dry-run: exiting without starting loop[/yellow]")
        raise typer.Exit(0)

    # Start the loop
    from orchestra_sdk.loop import ConductorLoop

    loop = ConductorLoop(cfg, resume=resume)
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Session state saved.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def status(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to conductor_config.yaml", exists=True
    ),
    env_file: Optional[Path] = typer.Option(None, "--env", "-e"),
) -> None:
    """Show the current status of a session from Supabase."""
    _load_env(env_file)

    from orchestra_sdk.config import ConductorConfig
    from orchestra_sdk.tools.supabase_tools import SupabaseClient

    cfg = ConductorConfig.from_yaml(config)
    client = SupabaseClient(cfg.supabase)

    session = client.get_session(cfg.session.name)
    if not session:
        console.print(f"[yellow]No session found for '{cfg.session.name}'[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Session: {cfg.session.name}", border_style="cyan")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for k, v in session.items():
        if k != "config":
            table.add_row(str(k), str(v))
    console.print(table)

    # Show last 5 experiments
    experiments = client.query_experiments(cfg.session.name, limit=5)
    if experiments:
        exp_table = Table(title="Last 5 Experiments", border_style="dim")
        exp_table.add_column("Iter", style="dim")
        exp_table.add_column("Hypothesis")
        exp_table.add_column("Metric")
        exp_table.add_column("Δ")
        exp_table.add_column("Decision")
        for exp in experiments:
            decision_color = "green" if exp.get("decision") == "keep" else "red"
            exp_table.add_row(
                str(exp.get("iteration", "?")),
                (exp.get("hypothesis", "")[:60] + "...") if len(exp.get("hypothesis", "")) > 60 else exp.get("hypothesis", ""),
                f"{exp.get('target_metric', 0):.4f}",
                f"{exp.get('delta', 0):+.4f}",
                f"[{decision_color}]{exp.get('decision', '?')}[/{decision_color}]",
            )
        console.print(exp_table)


@app.command()
def migrate(
    env_file: Optional[Path] = typer.Option(None, "--env", "-e"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print SQL without executing"),
) -> None:
    """Run database migrations to create Conductor tables in Supabase."""
    _load_env(env_file)

    from orchestra_sdk.db_migrations import run_migrations

    run_migrations(dry_run=dry_run, console=console)


@app.command()
def init(
    name: str = typer.Argument(..., help="Session name (e.g., memory_scribe_v1)"),
    output: Path = typer.Option(
        Path("conductor_config.yaml"), "--output", "-o"
    ),
) -> None:
    """Generate a starter conductor_config.yaml for a new session."""
    from orchestra_sdk.config_template import render_template

    content = render_template(name)
    output.write_text(content)
    console.print(f"[green]✓[/green] Created {output}")
    console.print(
        f"  Edit the file, then run: [cyan]orchestra run --config {output}[/cyan]"
    )


@app.command()
def inspect(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to conductor_config.yaml", exists=True
    ),
    env_file: Optional[Path] = typer.Option(None, "--env", "-e"),
    memories: bool = typer.Option(False, "--memories", "-m", help="Show memory contents"),
    fallback: bool = typer.Option(False, "--fallback", "-f", help="Replay JSONL fallback files"),
    git_log: bool = typer.Option(False, "--git-log", "-g", help="Show session git log"),
    best: bool = typer.Option(False, "--best", "-b", help="Show best model path and manifest"),
    all_: bool = typer.Option(False, "--all", "-a", help="Show everything (memories, fallback, git log, best)"),
) -> None:
    """Inspect session details: memories, JSONL fallback files, git log, best model."""
    _load_env(env_file)

    from orchestra_sdk.config import ConductorConfig
    from orchestra_sdk.tools.supabase_tools import SupabaseClient

    cfg = ConductorConfig.from_yaml(config)

    if all_:
        memories = fallback = git_log = best = True

    # ---- Best model path and manifest ----
    if best:
        best_dir = cfg.session.workspace_path / "best"
        manifest_path = best_dir / "best_manifest.json"
        if manifest_path.exists():
            import json as _json
            manifest = _json.loads(manifest_path.read_text())
            table = Table(title="Best Model", border_style="green")
            table.add_column("Field", style="cyan")
            table.add_column("Value")
            for k, v in manifest.items():
                table.add_row(str(k), str(v))
            console.print(table)
        elif best_dir.exists():
            console.print(f"[yellow]best/ directory exists at {best_dir} but no manifest found[/yellow]")
        else:
            console.print("[yellow]No best model saved yet (workspace/best/ not found)[/yellow]")

    # ---- JSONL fallback replay ----
    if fallback:
        fallback_dir = cfg.session.workspace_path / ".orchestra_fallback"
        if not fallback_dir.exists():
            console.print("[dim]No JSONL fallback files found (Supabase was available during all runs)[/dim]")
        else:
            import json as _json
            for jsonl_path in sorted(fallback_dir.glob("*.jsonl")):
                console.print(f"\n[cyan]Fallback file:[/cyan] {jsonl_path.name}")
                lines = jsonl_path.read_text().strip().splitlines()
                console.print(f"  {len(lines)} record(s)")
                fb_table = Table(border_style="dim")
                if lines:
                    first = _json.loads(lines[0])
                    for col in first.keys():
                        fb_table.add_column(str(col), overflow="fold")
                    for line in lines:
                        record = _json.loads(line)
                        fb_table.add_row(*[str(record.get(k, ""))[:60] for k in first.keys()])
                    console.print(fb_table)

    # ---- Memory contents ----
    if memories:
        try:
            from orchestra_sdk.memory.store import MemoryStore
            store = MemoryStore(cfg.supabase, cfg.memory, cfg.session.name)
            # Fetch all memories via a broad search
            results = store.search("iteration", top_k=50)
            if not results:
                console.print("[dim]No memories stored for this session[/dim]")
            else:
                mem_table = Table(title=f"Memories ({len(results)})", border_style="cyan")
                mem_table.add_column("Iter", style="dim", width=5)
                mem_table.add_column("Decision", width=8)
                mem_table.add_column("Content", overflow="fold")
                for m in results:
                    meta = m.get("metadata", {})
                    mem_table.add_row(
                        str(meta.get("iteration", "?")),
                        str(meta.get("decision", "?")),
                        str(m.get("content", ""))[:120],
                    )
                console.print(mem_table)
        except Exception as e:
            console.print(f"[red]Memory fetch failed:[/red] {e}")

    # ---- Session git log ----
    if git_log:
        from orchestra_sdk.tools.git_tools import GitManager
        gm = GitManager(cfg.session.workspace_path, cfg.session.branch)
        try:
            commits = gm.log.run(n=30)
            if not commits:
                console.print("[dim]No commits found in session workspace[/dim]")
            else:
                git_table = Table(title="Session Git Log (last 30)", border_style="dim")
                git_table.add_column("SHA", style="dim", width=9)
                git_table.add_column("Timestamp", width=20)
                git_table.add_column("Message", overflow="fold")
                for c in commits:
                    git_table.add_row(
                        c.short_sha,
                        c.timestamp.strftime("%Y-%m-%d %H:%M"),
                        c.message[:100],
                    )
                console.print(git_table)
        except Exception as e:
            console.print(f"[red]Git log failed:[/red] {e}")

    if not any([memories, fallback, git_log, best]):
        console.print(
            "[yellow]No flags specified. Use --memories, --fallback, --git-log, --best, or --all[/yellow]"
        )


@app.command()
def reset(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to conductor_config.yaml", exists=True
    ),
    to_iteration: int = typer.Option(
        ..., "--to-iteration", "-n", help="Revert session workspace to the git SHA from iteration N"
    ),
    env_file: Optional[Path] = typer.Option(None, "--env", "-e"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Revert the session workspace to the state at a specific iteration (hard reset)."""
    _load_env(env_file)

    from orchestra_sdk.config import ConductorConfig
    from orchestra_sdk.tools.supabase_tools import SupabaseClient
    from orchestra_sdk.tools.git_tools import GitManager

    cfg = ConductorConfig.from_yaml(config)
    client = SupabaseClient(cfg.supabase)
    gm = GitManager(cfg.session.workspace_path, cfg.session.branch)

    # Find the git SHA for the requested iteration from Supabase
    experiments = client.query_experiments(cfg.session.name, limit=500)
    target_exp = next(
        (e for e in experiments if e.get("iteration") == to_iteration), None
    )

    if not target_exp:
        console.print(
            f"[red]No experiment record found for iteration {to_iteration} "
            f"in session '{cfg.session.name}'[/red]"
        )
        raise typer.Exit(1)

    sha = target_exp.get("hypothesis_sha", "")
    if not sha:
        console.print(
            f"[red]Iteration {to_iteration} has no hypothesis_sha recorded — cannot reset[/red]"
        )
        raise typer.Exit(1)

    decision = target_exp.get("decision", "?")
    metric = target_exp.get("target_metric", "?")
    console.print(
        Panel(
            f"Session:    {cfg.session.name}\n"
            f"Reset to:   iteration {to_iteration} (decision={decision}, metric={metric})\n"
            f"Git SHA:    {sha[:12]}\n"
            f"[bold red]This will hard-reset the workspace. Uncommitted changes will be lost.[/bold red]",
            title="[yellow]orchestra reset[/yellow]",
            border_style="yellow",
        )
    )

    if not yes:
        confirmed = typer.confirm("Proceed with reset?")
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    try:
        result = gm.reset.run(sha)
        console.print(f"[green]✓ Workspace reset to {sha[:8]}[/green]")
    except Exception as e:
        console.print(f"[red]Git reset failed:[/red] {e}")
        raise typer.Exit(1)

    # Update Supabase session to reflect the reverted state
    baseline_at_iter = target_exp.get("target_metric")
    client.update_session(
        cfg.session.name,
        status="stopped",
        iteration=to_iteration,
        baseline_metric=baseline_at_iter,
    )

    # Delete orphaned experiment records for iterations after the reset point
    deleted = client.delete_experiments_after_iteration(cfg.session.name, to_iteration)
    orphan_note = " (orphaned experiment records cleaned)" if deleted else ""

    console.print(
        f"[green]✓ Session '{cfg.session.name}' updated: iteration={to_iteration}, "
        f"baseline_metric={baseline_at_iter}, status=stopped{orphan_note}[/green]"
    )
    console.print(
        f"  Resume from here with: [cyan]orchestra run --config {config} --resume[/cyan]"
    )


if __name__ == "__main__":
    app()
