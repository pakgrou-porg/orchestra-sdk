#!/usr/bin/env python3
"""
Orchestra SDK — Health Check & Validator
=========================================
Runs on Python 3.11+ with no third-party dependencies (stdlib only).
Works on Linux, macOS, and Windows.

Usage:
    python deploy/check.py [--env /path/to/.env] [--config /path/to/conductor_config.yaml]

Exit codes:
    0  All checks passed
    1  One or more checks failed
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Colour helpers (no third-party deps)
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"

_NO_COLOR = not sys.stdout.isatty() or bool(os.environ.get("NO_COLOR"))


def c(text: str, code: str) -> str:
    return text if _NO_COLOR else f"{code}{text}{RESET}"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class CheckResult(NamedTuple):
    name: str
    passed: bool
    detail: str
    critical: bool = True


results: list[CheckResult] = []


def record(name: str, passed: bool, detail: str, critical: bool = True) -> None:
    results.append(CheckResult(name, passed, detail, critical))
    icon = c("✓", GREEN) if passed else (c("✗", RED) if critical else c("⚠", YELLOW))
    print(f"  {icon}  {c(name, BOLD)}")
    if detail:
        print(f"       {c(detail, DIM)}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python() -> None:
    major, minor = sys.version_info[:2]
    passed = (major, minor) >= (3, 11)
    record("Python ≥ 3.11", passed, f"Found {sys.version.split()[0]}")


def check_binary(name: str, args: list[str], critical: bool = False) -> None:
    if not shutil.which(name):
        record(name, False, f"'{name}' not found in PATH", critical=critical)
        return
    try:
        r = subprocess.run([name, *args], capture_output=True, text=True, timeout=5)
        version_line = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "ok"
        record(name, True, version_line, critical=critical)
    except Exception as e:
        record(name, False, str(e), critical=critical)


def check_orchestra_package() -> None:
    spec = importlib.util.find_spec("orchestra_sdk")
    if spec is None:
        record("orchestra_sdk installed", False, "Package not found — run: pip install -e .")
        return
    try:
        import orchestra_sdk  # type: ignore
        version = getattr(orchestra_sdk, "__version__", "unknown")
        record("orchestra_sdk installed", True, f"version {version} at {spec.origin}")
    except Exception as e:
        record("orchestra_sdk installed", False, str(e))


def check_env_vars(env_file: Path | None) -> dict[str, str]:
    """Load .env and check required variables are present."""
    env: dict[str, str] = {}

    # Load from file
    if env_file and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
                os.environ.setdefault(k.strip(), v.strip())

    required = {
        "SUPABASE_URL": "Supabase project URL",
        "SUPABASE_ANON_KEY": "Supabase anon (public) key",
    }
    optional = {
        "SUPABASE_SERVICE_ROLE_KEY": "Supabase service role key (needed for migrations)",
        "OPENROUTER_API_KEY": "OpenRouter API key",
        "OPENAI_API_KEY": "OpenAI API key",
        "ANTHROPIC_API_KEY": "Anthropic API key",
        "OLLAMA_URL": "Ollama base URL",
    }

    for var, desc in required.items():
        val = os.environ.get(var, env.get(var, ""))
        record(f"ENV: {var}", bool(val), desc if val else f"NOT SET — {desc}", critical=True)

    for var, desc in optional.items():
        val = os.environ.get(var, env.get(var, ""))
        record(f"ENV: {var}", bool(val), desc if val else f"not set (optional)", critical=False)

    return env


def check_supabase(env: dict[str, str]) -> None:
    url = os.environ.get("SUPABASE_URL", env.get("SUPABASE_URL", ""))
    key = os.environ.get("SUPABASE_ANON_KEY", env.get("SUPABASE_ANON_KEY", ""))

    if not url or not key:
        record("Supabase reachable", False, "SUPABASE_URL or SUPABASE_ANON_KEY not set")
        return

    # Ping the REST endpoint
    try:
        req = urllib.request.Request(
            f"{url}/rest/v1/conductor_sessions?limit=1&select=id",
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            record("Supabase reachable", True, f"HTTP {resp.status} — conductor_sessions accessible")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 404:
            record("Supabase reachable", False,
                   "conductor_sessions table not found — run: orchestra migrate")
        else:
            record("Supabase reachable", False, f"HTTP {e.code}: {body[:120]}")
    except Exception as e:
        record("Supabase reachable", False, str(e))


def check_llm(env: dict[str, str]) -> None:
    """Try to reach the LLM provider's base URL."""
    providers = [
        ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/models", "OpenRouter"),
        ("OPENAI_API_KEY",     "https://api.openai.com/v1/models",    "OpenAI"),
        ("ANTHROPIC_API_KEY",  "https://api.anthropic.com/v1/models", "Anthropic"),
    ]
    found_any = False
    for key_var, url, name in providers:
        key = os.environ.get(key_var, env.get(key_var, ""))
        if not key:
            continue
        found_any = True
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                record(f"LLM: {name} reachable", True, f"HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            # 401 = key wrong, but server is reachable
            if e.code in (401, 403):
                record(f"LLM: {name} reachable", False,
                       f"HTTP {e.code} — API key may be invalid or expired")
            else:
                record(f"LLM: {name} reachable", True, f"HTTP {e.code} (server reachable)")
        except Exception as e:
            record(f"LLM: {name} reachable", False, str(e), critical=False)

    if not found_any:
        record("LLM provider", False,
               "No LLM API key found. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY",
               critical=False)


def check_ollama(env: dict[str, str]) -> None:
    url = os.environ.get("OLLAMA_URL", env.get("OLLAMA_URL", "http://localhost:11434"))
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            has_embed = any("nomic-embed-text" in m for m in models)
            detail = f"{len(models)} model(s) loaded"
            if has_embed:
                detail += " — nomic-embed-text ✓"
            else:
                detail += " — nomic-embed-text NOT found (run: ollama pull nomic-embed-text)"
            record("Ollama reachable", True, detail, critical=False)
    except Exception:
        record("Ollama reachable", False,
               f"Cannot reach {url} — memory embeddings disabled. "
               "Install from https://ollama.com", critical=False)


def check_docker() -> None:
    if not shutil.which("docker"):
        record("Docker", False, "Not installed — 'docker' runner unavailable", critical=False)
        return
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            record("Docker daemon", True, "Running")
        else:
            record("Docker daemon", False,
                   "Docker is installed but daemon is not running", critical=False)
    except Exception as e:
        record("Docker daemon", False, str(e), critical=False)


def check_conductor_image() -> None:
    if not shutil.which("docker"):
        return
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", "orchestra-musician:latest"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            record("Docker image: orchestra-musician:latest", True, "Image present")
        else:
            record("Docker image: orchestra-musician:latest", False,
                   "Not built — run: docker build -f docker/Dockerfile.musician "
                   "-t orchestra-musician:latest .", critical=False)
    except Exception as e:
        record("Docker image: orchestra-musician:latest", False, str(e), critical=False)


def check_config(config_path: Path | None) -> None:
    if config_path is None:
        default = Path("conductor_config.yaml")
        if default.exists():
            config_path = default
        else:
            record("conductor_config.yaml", False,
                   "Not found. Run setup.py or copy an example from examples/")
            return

    if not config_path.exists():
        record("conductor_config.yaml", False, f"File not found: {config_path}")
        return

    try:
        import yaml  # type: ignore
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        session = raw.get("session", {})
        llm = raw.get("llm", {})
        missing = []
        for field in ("name", "dataset_id", "workspace_dir", "branch"):
            if not session.get(field):
                missing.append(f"session.{field}")
        for field in ("provider", "model", "api_key_env"):
            if not llm.get(field):
                missing.append(f"llm.{field}")
        if missing:
            record("conductor_config.yaml valid", False,
                   "Missing required fields: " + ", ".join(missing))
        else:
            record("conductor_config.yaml valid", True,
                   f"session={session['name']}, llm={llm['provider']}/{llm['model']}")
    except ImportError:
        record("conductor_config.yaml valid", False,
               "PyYAML not installed — cannot validate config")
    except Exception as e:
        record("conductor_config.yaml valid", False, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orchestra SDK health check")
    p.add_argument("--env", "-e", type=Path, default=None,
                   help="Path to .env file (default: .env in repo root)")
    p.add_argument("--config", "-c", type=Path, default=None,
                   help="Path to conductor_config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    env_file = args.env or (repo_root / ".env")
    config_path = args.config

    print()
    print(c("╔══════════════════════════════════════════════════════════╗", CYAN))
    print(c("║         Orchestra SDK — Health Check & Validator         ║", BOLD + CYAN))
    print(c("╚══════════════════════════════════════════════════════════╝", CYAN))
    print()
    print(c(f"  OS:      {platform.system()} {platform.release()}", DIM))
    print(c(f"  Python:  {sys.version.split()[0]}", DIM))
    print(c(f"  Repo:    {repo_root}", DIM))
    print(c(f"  Env:     {env_file}", DIM))
    print()

    print(c("── System binaries ─────────────────────────────────────", DIM))
    check_python()
    check_binary("git", ["--version"], critical=True)
    check_binary("docker", ["--version"], critical=False)
    check_binary("ollama", ["--version"], critical=False)
    check_binary("kubectl", ["version", "--client", "--short"], critical=False)

    print()
    print(c("── Python package ──────────────────────────────────────", DIM))
    check_orchestra_package()

    print()
    print(c("── Environment variables ───────────────────────────────", DIM))
    env = check_env_vars(env_file)

    print()
    print(c("── Connectivity ────────────────────────────────────────", DIM))
    check_supabase(env)
    check_llm(env)
    check_ollama(env)

    print()
    print(c("── Docker ──────────────────────────────────────────────", DIM))
    check_docker()
    check_conductor_image()

    print()
    print(c("── Configuration ───────────────────────────────────────", DIM))
    check_config(config_path)

    # Summary
    passed = [r for r in results if r.passed]
    failed_critical = [r for r in results if not r.passed and r.critical]
    failed_optional = [r for r in results if not r.passed and not r.critical]

    print()
    print(c("── Summary ─────────────────────────────────────────────", DIM))
    print(f"  {c(str(len(passed)), GREEN)} passed  "
          f"{c(str(len(failed_critical)), RED)} critical failures  "
          f"{c(str(len(failed_optional)), YELLOW)} warnings")

    if failed_critical:
        print()
        print(c("  Critical issues to fix:", RED + BOLD))
        for r in failed_critical:
            print(c(f"    • {r.name}: {r.detail}", RED))

    if failed_optional:
        print()
        print(c("  Optional / non-blocking:", YELLOW))
        for r in failed_optional:
            print(c(f"    • {r.name}: {r.detail}", YELLOW))

    print()
    if failed_critical:
        print(c("  ✗  System is NOT ready. Fix critical issues above.", RED + BOLD))
        sys.exit(1)
    else:
        print(c("  ✓  System is ready to run Orchestra.", GREEN + BOLD))
        sys.exit(0)


if __name__ == "__main__":
    main()
