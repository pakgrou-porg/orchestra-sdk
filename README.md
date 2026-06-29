# Orchestra SDK

Autonomous LLM-driven research loop for iterative LLM fine-tuning. The Conductor runs a closed loop — proposing hypotheses, applying edits to training code, running experiments, and keeping only improvements — entirely without human intervention.

**Dashboard:** [orchdash-h5l46992.manus.space](https://orchdash-h5l46992.manus.space)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Deployment & Setup](#deployment--setup)
4. [Configuration Reference](#configuration-reference)
5. [CLI Reference](#cli-reference)
6. [Running Tests](#running-tests)
7. [Examples](#examples)

---

## Architecture

The Conductor executes a 10-step loop on every iteration:

| Step | Action |
|------|--------|
| 1 | Read `program.md` — research goals and constraints |
| 2 | Search semantic memories (pgvector retrieval via Supabase) |
| 3 | Read git log — recent changes to the training script |
| 4 | Read metric history from Supabase |
| 5 | Propose a hypothesis (LLM structured output → `Hypothesis` schema) |
| 6 | Apply edit to `train.py` (find/replace patch) |
| 7 | Commit the candidate to git |
| 8 | Run the experiment (Docker, Kubernetes, or local runner) |
| 9 | Read `results.json` produced by the training script |
| 10 | Keep or discard based on `keep_threshold`; update Supabase |

The SDK has four runner modes — `local` (direct subprocess), `docker` (per-iteration container with GPU passthrough), `k8s` (Kubernetes Job), and a synthetic no-GPU mode for testing.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.11 | [python.org](https://www.python.org/downloads/) |
| Git | any | [git-scm.com](https://git-scm.com/) |
| Docker | ≥ 24 | Optional — required for `runner.type: docker` |
| NVIDIA drivers + CUDA | ≥ 12.1 | Optional — required for GPU training |
| Ollama | any | Optional — required for semantic memory embeddings |
| kubectl | ≥ 1.29 | Optional — required for `runner.type: k8s` |

---

## Deployment & Setup

The `deploy/` directory contains a complete cross-platform toolkit for installing and configuring the SDK on a fresh machine. See [`deploy/README.md`](deploy/README.md) for the full reference.

### Linux / macOS

```bash
# 1. Clone the repository
git clone https://github.com/pakgrou-porg/orchestra-sdk.git
cd orchestra-sdk

# 2. Run the interactive setup wizard
#    Installs dependencies, collects secrets, writes .env and
#    conductor_config.yaml, runs migrations, and verifies the stack.
python deploy/setup.py

# 3. Make the launchers executable
chmod +x deploy/orchestra.sh deploy/build_musician.sh deploy/migrate.sh

# 4. Verify everything is wired up
./deploy/orchestra.sh check

# 5. Start a session
./deploy/orchestra.sh run --config conductor_config.yaml
```

### Windows (PowerShell)

```powershell
# Allow local scripts (run once as Administrator if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

git clone https://github.com/pakgrou-porg/orchestra-sdk.git
cd orchestra-sdk

python deploy\setup.py

.\deploy\orchestra.ps1 check
.\deploy\orchestra.ps1 run --config conductor_config.yaml
```

### Windows (CMD)

```cmd
git clone https://github.com/pakgrou-porg/orchestra-sdk.git
cd orchestra-sdk
python deploy\setup.py
deploy\orchestra.bat check
deploy\orchestra.bat run --config conductor_config.yaml
```

### What the Setup Wizard Configures

Running `python deploy/setup.py` walks through eight automated steps:

1. **Prerequisites check** — Python version, Git, Docker, Ollama, kubectl
2. **Virtual environment** — creates `.venv` in the repo root
3. **Dependency installation** — installs `orchestra_sdk` (editable) and all runtime packages
4. **Secrets collection** — prompts for Supabase URL/keys and LLM provider API key; pre-fills from any existing `.env`
5. **File generation** — writes `.env` and a starter `conductor_config.yaml`
6. **Database migrations** — runs `orchestra migrate` to create all Supabase tables and RLS policies
7. **Ollama model pull** — pulls `nomic-embed-text` if Ollama is installed
8. **Self-test** — runs `orchestra status` to confirm the full stack is operational

### Secrets

All secrets live in `.env` at the repository root (excluded from version control). Copy `deploy/.env.example` as a starting point:

```bash
cp deploy/.env.example .env
# then edit .env with your values
```

| Variable | Required | Where to find it |
|----------|----------|-----------------|
| `SUPABASE_URL` | Yes | Supabase dashboard → Project Settings → API |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase dashboard → Project Settings → API — used by the trusted Conductor at runtime and for migrations (backend-only) |
| `SUPABASE_ANON_KEY` | Optional | Supabase dashboard → Project Settings → API — read-only dashboard / runtime fallback only |
| `OPENROUTER_API_KEY` | One LLM key required | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `OPENAI_API_KEY` | One LLM key required | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | One LLM key required | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `OLLAMA_URL` | No | Default: `http://localhost:11434` |

### Building the Musician Docker Image

The Musician container is the training environment launched per iteration when `runner.type: docker`.

```bash
# NVIDIA CUDA image (requires NVIDIA drivers + Docker)
./deploy/build_musician.sh

# CPU-only image (for testing without a GPU)
./deploy/build_musician.sh --cpu

# Windows
deploy\build_musician.bat
deploy\build_musician.bat --cpu
```

### Database Migrations

```bash
# Linux / macOS
./deploy/migrate.sh --config conductor_config.yaml

# Windows
deploy\migrate.bat --config conductor_config.yaml
```

Migrations require `SUPABASE_SERVICE_ROLE_KEY`. They are idempotent and safe to re-run after SDK updates.

### Health Check

```bash
./deploy/orchestra.sh check        # Linux / macOS
.\deploy\orchestra.ps1 check       # Windows PowerShell
deploy\orchestra.bat check         # Windows CMD
```

Exits `0` if all critical checks pass, `1` otherwise — suitable for CI pipelines.

---

## Configuration Reference

The full `conductor_config.yaml` field reference is in [`deploy/README.md`](deploy/README.md#conductor_configyaml-reference). A minimal working example:

```yaml
session:
  name: "my_session_v1"
  dataset_id: "my_dataset"
  workspace_dir: "~/.orchestra/workspace/my_session_v1"
  branch: "session/my_session_v1"
  max_iterations: 50
  target_metric: "val_loss"
  keep_threshold: -0.005

llm:
  provider: "openrouter"
  model: "anthropic/claude-3-5-haiku-20241022"
  api_key_env: "OPENROUTER_API_KEY"

runner:
  type: "local"       # or "docker" / "k8s"
  gpu_device: "none"

memory:
  enabled: false      # set true once Ollama is running

supabase:
  url_env: "SUPABASE_URL"
  key_env: "SUPABASE_SERVICE_ROLE_KEY"   # trusted backend; anon key is read-only under hardened RLS
```

See `examples/memory_scribe/conductor_config.yaml` and `examples/synthetic/conductor_config.yaml` for complete annotated examples.

---

## CLI Reference

All commands accept `--env /path/to/.env` to load secrets from a custom location.

```
orchestra init    --config FILE          Initialise a new session in Supabase
orchestra run     --config FILE          Start (or resume) a Conductor session
orchestra status  --config FILE [flags]  Show session status, memories, git log
orchestra migrate --config FILE          Apply database migrations
orchestra inspect --config FILE [flags]  Inspect memories, fallback history, best run
orchestra reset   --config FILE -n N     Revert workspace to iteration N
```

The launcher scripts additionally expose:

```
setup    Run the interactive setup wizard
check    Run the health-check validator
```

### Rollback

To revert a session workspace to a previous iteration:

```bash
# Find the iteration number
orchestra inspect --config conductor_config.yaml --git-log

# Revert to iteration 12
orchestra reset --config conductor_config.yaml --to-iteration 12

# Resume from the reverted state
orchestra run --config conductor_config.yaml --resume
```

---

## Running Tests

```bash
# Activate the virtual environment first
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pytest tests/ -v
```

The test suite includes unit tests (`tests/test_config_and_context.py`, `tests/test_tools.py`) and an integration test (`tests/test_integration.py`) that exercises the full Conductor loop against the synthetic example.

---

## Examples

| Example | Description |
|---------|-------------|
| `examples/memory_scribe/` | Full GPU session with Ollama memory, OpenRouter LLM, Docker runner |
| `examples/synthetic/` | No-GPU end-to-end test using the local runner and a synthetic training script |

Each example directory contains a `conductor_config.yaml` and a `train.py` that can be used as a starting point for new sessions.
