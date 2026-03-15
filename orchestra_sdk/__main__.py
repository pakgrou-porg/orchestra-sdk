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


if __name__ == "__main__":
    app()
