#!/usr/bin/env python3
"""
Orchestra SDK — Interactive Setup Wizard
========================================
Runs on Python 3.11+ with no third-party dependencies.
Works on Linux, macOS, and Windows (PowerShell or CMD).

Usage:
    python deploy/setup.py

What it does:
  1. Checks Python version and OS prerequisites
  2. Creates / activates a virtual environment
  3. Installs the orchestra_sdk package and all dependencies
  4. Interactively collects secrets (Supabase URL/key, LLM API key, etc.)
  5. Writes a .env file and a starter conductor_config.yaml
  6. Runs database migrations (orchestra migrate)
  7. Pulls the Ollama embedding model (if Ollama is detected)
  8. Runs a self-test (orchestra status) to confirm everything is wired up
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"

_NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")


def c(text: str, code: str) -> str:
    return text if _NO_COLOR else f"{code}{text}{RESET}"


def header(msg: str) -> None:
    width = 60
    print()
    print(c("─" * width, DIM))
    print(c(f"  {msg}", BOLD + CYAN))
    print(c("─" * width, DIM))


def ok(msg: str) -> None:
    print(c(f"  ✓  {msg}", GREEN))


def warn(msg: str) -> None:
    print(c(f"  ⚠  {msg}", YELLOW))


def err(msg: str) -> None:
    print(c(f"  ✗  {msg}", RED))


def info(msg: str) -> None:
    print(c(f"     {msg}", DIM))


def ask(prompt: str, default: str = "", secret: bool = False) -> str:
    """Prompt the user for input, with an optional default."""
    display_default = "****" if (secret and default) else default
    suffix = f" [{display_default}]" if default else ""
    full_prompt = c(f"  → {prompt}{suffix}: ", BOLD)
    if secret:
        import getpass
        value = getpass.getpass(full_prompt) or default
    else:
        value = input(full_prompt).strip() or default
    return value


def run(cmd: list[str], check: bool = True, capture: bool = False, **kwargs):
    """Run a subprocess command."""
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, **kwargs)
    return subprocess.run(cmd, check=check, **kwargs)


IS_WINDOWS = platform.system() == "Windows"
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Step 1 — Prerequisites
# ---------------------------------------------------------------------------

def check_prerequisites() -> None:
    header("Step 1 / 8 — Checking prerequisites")

    # Python version
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        err(f"Python 3.11+ required (found {major}.{minor}). Aborting.")
        sys.exit(1)
    ok(f"Python {major}.{minor}")

    # Git
    if shutil.which("git"):
        result = run(["git", "--version"], capture=True, check=False)
        ok(result.stdout.strip())
    else:
        err("git not found. Install git and re-run.")
        sys.exit(1)

    # Docker (optional)
    if shutil.which("docker"):
        result = run(["docker", "--version"], capture=True, check=False)
        ok(result.stdout.strip() + " (Docker available)")
    else:
        warn("Docker not found — 'docker' runner type will not work. "
             "Use runner.type: local or k8s.")

    # Ollama (optional)
    if shutil.which("ollama"):
        result = run(["ollama", "--version"], capture=True, check=False)
        ok(result.stdout.strip() + " (Ollama available — memory embeddings enabled)")
    else:
        warn("Ollama not found — memory embeddings will be disabled. "
             "Install from https://ollama.com to enable semantic memory.")

    # kubectl (optional)
    if shutil.which("kubectl"):
        result = run(["kubectl", "version", "--client", "--short"], capture=True, check=False)
        ok(result.stdout.strip() + " (kubectl available)")
    else:
        info("kubectl not found — Kubernetes runner will not be available.")


# ---------------------------------------------------------------------------
# Step 2 — Virtual environment
# ---------------------------------------------------------------------------

def setup_venv() -> Path:
    header("Step 2 / 8 — Virtual environment")

    venv_dir = REPO_ROOT / ".venv"
    python_bin = venv_dir / ("Scripts" if IS_WINDOWS else "bin") / ("python.exe" if IS_WINDOWS else "python")

    if venv_dir.exists():
        ok(f"Virtual environment already exists at {venv_dir}")
    else:
        info(f"Creating virtual environment at {venv_dir} …")
        run([sys.executable, "-m", "venv", str(venv_dir)])
        ok(f"Created virtual environment at {venv_dir}")

    return python_bin


# ---------------------------------------------------------------------------
# Step 3 — Install dependencies
# ---------------------------------------------------------------------------

def install_dependencies(python_bin: Path) -> None:
    header("Step 3 / 8 — Installing orchestra_sdk and dependencies")

    pip = [str(python_bin), "-m", "pip"]
    info("Upgrading pip …")
    run([*pip, "install", "--upgrade", "pip", "--quiet"])

    info("Installing orchestra_sdk (editable) …")
    run([*pip, "install", "-e", str(REPO_ROOT), "--quiet"])
    ok("orchestra_sdk installed")

    # Verify CLI is available
    orchestra_bin = python_bin.parent / ("orchestra.exe" if IS_WINDOWS else "orchestra")
    if orchestra_bin.exists():
        ok(f"CLI available at {orchestra_bin}")
    else:
        warn("'orchestra' CLI not found in venv bin — check PATH after activation.")


# ---------------------------------------------------------------------------
# Step 4 — Collect secrets
# ---------------------------------------------------------------------------

LLM_PROVIDERS = {
    "1": ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "2": ("openai",     "OPENAI_API_KEY",     "https://api.openai.com/v1"),
    "3": ("anthropic",  "ANTHROPIC_API_KEY",  "https://api.anthropic.com/v1"),
    "4": ("lmstudio",   "",                   "http://localhost:1234/v1"),
    "5": ("custom",     "LLM_API_KEY",        ""),
}

LLM_MODELS = {
    "openrouter": "anthropic/claude-3-5-haiku-20241022",
    "openai":     "gpt-4o-mini",
    "anthropic":  "claude-3-5-haiku-20241022",
    "lmstudio":   "local-model",
    "custom":     "your-model-name",
}


def collect_secrets(existing_env: dict[str, str]) -> dict[str, str]:
    header("Step 4 / 8 — Secrets & configuration")

    secrets: dict[str, str] = dict(existing_env)

    print()
    print(c("  Supabase", BOLD))
    print(c("  Find these at: https://supabase.com/dashboard/project/_/settings/api", DIM))
    secrets["SUPABASE_URL"] = ask(
        "Supabase project URL",
        secrets.get("SUPABASE_URL", "https://<ref>.supabase.co"),
    )
    secrets["SUPABASE_ANON_KEY"] = ask(
        "Supabase anon key",
        secrets.get("SUPABASE_ANON_KEY", ""),
        secret=True,
    )
    secrets["SUPABASE_SERVICE_ROLE_KEY"] = ask(
        "Supabase service role key (for migrations)",
        secrets.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        secret=True,
    )

    print()
    print(c("  LLM Provider", BOLD))
    print("    1) OpenRouter (recommended — access to Claude, GPT-4, Mistral, etc.)")
    print("    2) OpenAI")
    print("    3) Anthropic (direct)")
    print("    4) LM Studio (local, no key required)")
    print("    5) Custom / self-hosted")
    choice = ask("Choose provider", "1")
    provider, key_env, base_url = LLM_PROVIDERS.get(choice, LLM_PROVIDERS["1"])

    if key_env:
        secrets[key_env] = ask(
            f"{provider} API key ({key_env})",
            secrets.get(key_env, ""),
            secret=True,
        )
    secrets["_LLM_PROVIDER"] = provider
    secrets["_LLM_KEY_ENV"] = key_env
    secrets["_LLM_BASE_URL"] = base_url

    print()
    print(c("  Ollama (memory embeddings)", BOLD))
    secrets["OLLAMA_URL"] = ask(
        "Ollama base URL",
        secrets.get("OLLAMA_URL", "http://localhost:11434"),
    )

    return secrets


# ---------------------------------------------------------------------------
# Step 5 — Write .env and conductor_config.yaml
# ---------------------------------------------------------------------------

def write_env_file(secrets: dict[str, str]) -> Path:
    env_path = REPO_ROOT / ".env"

    lines = [
        "# Orchestra SDK — environment secrets",
        "# Generated by deploy/setup.py — DO NOT commit this file",
        "",
        "# Supabase",
        f"SUPABASE_URL={secrets.get('SUPABASE_URL', '')}",
        f"SUPABASE_ANON_KEY={secrets.get('SUPABASE_ANON_KEY', '')}",
        f"SUPABASE_SERVICE_ROLE_KEY={secrets.get('SUPABASE_SERVICE_ROLE_KEY', '')}",
        "",
        "# LLM provider API key",
    ]

    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY"):
        if secrets.get(key):
            lines.append(f"{key}={secrets[key]}")

    lines += [
        "",
        "# Ollama",
        f"OLLAMA_URL={secrets.get('OLLAMA_URL', 'http://localhost:11434')}",
    ]

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def write_conductor_config(secrets: dict[str, str], session_name: str) -> Path:
    provider = secrets.get("_LLM_PROVIDER", "openrouter")
    key_env = secrets.get("_LLM_KEY_ENV", "OPENROUTER_API_KEY")
    model = LLM_MODELS.get(provider, "anthropic/claude-3-5-haiku-20241022")
    base_url_line = ""
    if secrets.get("_LLM_BASE_URL") and provider not in ("openrouter", "openai", "anthropic"):
        base_url_line = f"  base_url: \"{secrets['_LLM_BASE_URL']}\"\n"

    config_path = REPO_ROOT / "conductor_config.yaml"
    content = textwrap.dedent(f"""\
        # Orchestra Conductor Configuration
        # Generated by deploy/setup.py
        # Edit this file to customise your session.

        session:
          name: "{session_name}"
          dataset_id: "{session_name}"
          workspace_dir: "~/.orchestra/workspace/{session_name}"
          branch: "session/{session_name}"
          max_iterations: 50
          target_metric: "val_loss"
          target_value: 2.30
          keep_threshold: -0.005

        llm:
          provider: "{provider}"
          model: "{model}"
          api_key_env: "{key_env}"
        {base_url_line}  max_tokens: 4096
          temperature: 0.3
          context_budget_tokens: 8000

        runner:
          type: "local"          # Change to "docker" or "k8s" as needed
          gpu_device: "0"        # Set to "none" for CPU-only
          timeout_seconds: 3600

        memory:
          embedding_model: "nomic-embed-text"
          embedding_url: "{secrets.get('OLLAMA_URL', 'http://localhost:11434')}"
          top_k: 5
          similarity_threshold: 0.75
          enabled: false         # Set to true once Ollama is running

        supabase:
          url_env: "SUPABASE_URL"
          key_env: "SUPABASE_ANON_KEY"

        program:
          path: "program.md"
          train_script: "train.py"
          eval_script: "evaluate.py"
          results_file: "results.json"
          datasets_dir: "~/.orchestra/datasets"
        """)
    config_path.write_text(content, encoding="utf-8")
    return config_path


def write_files(secrets: dict[str, str]) -> tuple[Path, Path]:
    header("Step 5 / 8 — Writing .env and conductor_config.yaml")

    session_name = ask("Session name (used as workspace and dataset identifier)", "my_session_v1")

    env_path = write_env_file(secrets)
    ok(f"Wrote {env_path}")

    config_path = write_conductor_config(secrets, session_name)
    ok(f"Wrote {config_path}")

    warn(".env is in .gitignore — never commit it.")
    return env_path, config_path


# ---------------------------------------------------------------------------
# Step 6 — Run migrations
# ---------------------------------------------------------------------------

def run_migrations(python_bin: Path, config_path: Path, env_path: Path) -> None:
    header("Step 6 / 8 — Running database migrations")

    orchestra_bin = python_bin.parent / ("orchestra.exe" if IS_WINDOWS else "orchestra")
    cmd = [str(orchestra_bin), "migrate", "--config", str(config_path), "--env", str(env_path)]

    info(f"Running: {' '.join(cmd)}")
    result = run(cmd, check=False)
    if result.returncode == 0:
        ok("Migrations applied successfully")
    else:
        warn("Migration command returned non-zero. Check output above.")
        warn("You can re-run migrations later with: orchestra migrate --config conductor_config.yaml")


# ---------------------------------------------------------------------------
# Step 7 — Pull Ollama model
# ---------------------------------------------------------------------------

def pull_ollama_model(secrets: dict[str, str]) -> None:
    header("Step 7 / 8 — Ollama embedding model")

    if not shutil.which("ollama"):
        warn("Ollama not installed — skipping model pull.")
        info("Install from https://ollama.com, then run: ollama pull nomic-embed-text")
        return

    model = "nomic-embed-text"
    info(f"Pulling {model} …")
    result = run(["ollama", "pull", model], check=False)
    if result.returncode == 0:
        ok(f"Model '{model}' ready")
    else:
        warn(f"Failed to pull '{model}'. Run manually: ollama pull {model}")


# ---------------------------------------------------------------------------
# Step 8 — Self-test
# ---------------------------------------------------------------------------

def self_test(python_bin: Path, config_path: Path, env_path: Path) -> None:
    header("Step 8 / 8 — Self-test")

    orchestra_bin = python_bin.parent / ("orchestra.exe" if IS_WINDOWS else "orchestra")
    cmd = [str(orchestra_bin), "status", "--config", str(config_path), "--env", str(env_path)]

    info(f"Running: {' '.join(cmd)}")
    result = run(cmd, check=False)
    if result.returncode == 0:
        ok("orchestra status passed — setup complete!")
    else:
        warn("Status check returned non-zero. Review the output above.")
        warn("Common causes: missing env vars, Supabase unreachable, or no session yet created.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_existing_env() -> dict[str, str]:
    """Load any existing .env file so we can pre-fill prompts."""
    env_path = REPO_ROOT / ".env"
    result: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip()
    return result


def main() -> None:
    print()
    print(c("╔══════════════════════════════════════════════════════════╗", CYAN))
    print(c("║         Orchestra SDK — Interactive Setup Wizard         ║", BOLD + CYAN))
    print(c("╚══════════════════════════════════════════════════════════╝", CYAN))
    print()
    print(c("  This wizard will install and configure the Orchestra SDK.", DIM))
    print(c("  It will create a .env file and a conductor_config.yaml.", DIM))
    print(c("  Press Ctrl+C at any time to abort.", DIM))
    print()

    try:
        check_prerequisites()
        python_bin = setup_venv()
        install_dependencies(python_bin)
        existing = load_existing_env()
        secrets = collect_secrets(existing)
        env_path, config_path = write_files(secrets)
        run_migrations(python_bin, config_path, env_path)
        pull_ollama_model(secrets)
        self_test(python_bin, config_path, env_path)
    except KeyboardInterrupt:
        print()
        warn("Setup aborted by user.")
        sys.exit(1)

    print()
    print(c("══════════════════════════════════════════════════════════", CYAN))
    print(c("  Setup complete! Next steps:", BOLD + GREEN))
    print()
    print(c("  1. Activate the virtual environment:", DIM))
    if IS_WINDOWS:
        print(c("       .venv\\Scripts\\activate", CYAN))
    else:
        print(c("       source .venv/bin/activate", CYAN))
    print()
    print(c("  2. Edit conductor_config.yaml to point at your workspace:", DIM))
    print(c("       session.workspace_dir, program.train_script, etc.", CYAN))
    print()
    print(c("  3. Start a session:", DIM))
    print(c("       orchestra run --config conductor_config.yaml", CYAN))
    print()
    print(c("  4. Monitor progress:", DIM))
    print(c("       orchestra status --config conductor_config.yaml", CYAN))
    print(c("       https://orchdash-h5l46992.manus.space", CYAN))
    print(c("══════════════════════════════════════════════════════════", CYAN))
    print()


if __name__ == "__main__":
    main()
