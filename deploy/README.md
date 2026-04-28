# Orchestra SDK — Deployment & Configuration Toolkit

This directory contains everything needed to install, configure, and operate the Orchestra SDK on a fresh Linux, macOS, or Windows machine. The toolkit is designed to be run from the repository root and requires only Python 3.11 and Git as prerequisites.

---

## Contents

| File | Purpose |
|------|---------|
| `setup.py` | Interactive setup wizard — installs dependencies, collects secrets, writes `.env` and `conductor_config.yaml`, runs migrations, and pulls the Ollama embedding model |
| `check.py` | Health-check validator — verifies Python version, binaries, env vars, Supabase connectivity, LLM reachability, Ollama status, Docker, and config validity |
| `.env.example` | Annotated secrets template — copy to `.env` and fill in values |
| `orchestra.sh` | Linux / macOS launcher — activates venv, loads `.env`, dispatches to the `orchestra` CLI |
| `orchestra.bat` | Windows CMD launcher — same as above for Command Prompt |
| `orchestra.ps1` | Windows PowerShell launcher — same as above for PowerShell |
| `build_musician.sh` | Linux / macOS Docker image builder for the Musician training container |
| `build_musician.bat` | Windows Docker image builder |
| `migrate.sh` | Linux / macOS migration runner (requires `SUPABASE_SERVICE_ROLE_KEY`) |
| `migrate.bat` | Windows migration runner |

---

## Prerequisites

The following must be installed before running the setup wizard. Items marked **optional** are only needed for specific runner modes.

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.11 | [python.org](https://www.python.org/downloads/) |
| Git | any | [git-scm.com](https://git-scm.com/) |
| Docker | ≥ 24 | **Optional** — required for `runner.type: docker` |
| NVIDIA drivers + CUDA | ≥ 12.1 | **Optional** — required for GPU training |
| Ollama | any | **Optional** — required for semantic memory embeddings |
| kubectl | ≥ 1.29 | **Optional** — required for `runner.type: k8s` |

---

## Quick Start

### Linux / macOS

```bash
# 1. Clone the repository
git clone https://github.com/pakgrou-porg/orchestra-sdk.git
cd orchestra-sdk

# 2. Run the interactive setup wizard
python deploy/setup.py

# 3. Make the launcher executable
chmod +x deploy/orchestra.sh deploy/build_musician.sh deploy/migrate.sh

# 4. Start a session
./deploy/orchestra.sh run --config conductor_config.yaml
```

### Windows (PowerShell)

```powershell
# Allow local scripts (run once as Administrator if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 1. Clone the repository
git clone https://github.com/pakgrou-porg/orchestra-sdk.git
cd orchestra-sdk

# 2. Run the interactive setup wizard
python deploy\setup.py

# 3. Start a session
.\deploy\orchestra.ps1 run --config conductor_config.yaml
```

### Windows (CMD)

```cmd
git clone https://github.com/pakgrou-porg/orchestra-sdk.git
cd orchestra-sdk
python deploy\setup.py
deploy\orchestra.bat run --config conductor_config.yaml
```

---

## What the Setup Wizard Does

Running `python deploy/setup.py` walks through eight steps automatically:

1. **Prerequisites check** — verifies Python version, Git, Docker, Ollama, and kubectl.
2. **Virtual environment** — creates `.venv` in the repo root (skips if already present).
3. **Dependency installation** — installs `orchestra_sdk` in editable mode plus all runtime dependencies from `pyproject.toml`.
4. **Secrets collection** — interactively prompts for Supabase URL/keys and LLM provider API key. Existing `.env` values are pre-filled so re-runs are non-destructive.
5. **File generation** — writes `.env` and a starter `conductor_config.yaml` tailored to the chosen LLM provider.
6. **Database migrations** — runs `orchestra migrate` to create all Supabase tables and RLS policies.
7. **Ollama model pull** — pulls `nomic-embed-text` if Ollama is installed.
8. **Self-test** — runs `orchestra status` to confirm the full stack is wired up.

---

## Secrets Reference

All secrets are stored in `.env` at the repository root. The file is excluded from version control via `.gitignore`. Use `deploy/.env.example` as a starting template.

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | **Yes** | Supabase project URL, e.g. `https://abc123.supabase.co` |
| `SUPABASE_ANON_KEY` | **Yes** | Supabase anon (public) key — used by the Conductor at runtime |
| `SUPABASE_SERVICE_ROLE_KEY` | Migrations only | Supabase service role key — used by `orchestra migrate` for DDL |
| `OPENROUTER_API_KEY` | One LLM key required | OpenRouter key — access to Claude, GPT-4, Mistral, Llama, etc. |
| `OPENAI_API_KEY` | One LLM key required | OpenAI direct key |
| `ANTHROPIC_API_KEY` | One LLM key required | Anthropic direct key |
| `OLLAMA_URL` | No | Ollama base URL (default: `http://localhost:11434`) |

Supabase keys are found at **Project Settings → API** in the [Supabase dashboard](https://supabase.com/dashboard).

---

## `conductor_config.yaml` Reference

The setup wizard generates a starter config. The table below describes every field.

### `session`

| Field | Default | Description |
|-------|---------|-------------|
| `name` | — | Unique session identifier used as the Supabase key and git branch prefix |
| `dataset_id` | — | Dataset identifier matching a row in the `datasets` Supabase table |
| `workspace_dir` | `~/.orchestra/workspace/<name>` | Local path to the training code repository |
| `branch` | `session/<name>` | Git branch the Conductor commits to |
| `max_iterations` | 50 | Maximum number of Conductor iterations before stopping |
| `target_metric` | `val_loss` | Metric name returned in `results.json` |
| `target_value` | null | Stop early if metric reaches this value |
| `keep_threshold` | -0.005 | Keep a commit only if the metric delta is ≤ this value |

### `llm`

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `openrouter` | One of `openrouter`, `openai`, `anthropic`, `lmstudio`, or `custom` |
| `model` | — | Model identifier as accepted by the provider's API |
| `api_key_env` | `OPENROUTER_API_KEY` | Name of the environment variable holding the API key |
| `base_url` | (provider default) | Override the API base URL for custom or self-hosted endpoints |
| `max_tokens` | 4096 | Maximum tokens per LLM response |
| `temperature` | 0.3 | Sampling temperature |
| `context_budget_tokens` | 8000 | Maximum tokens of context fed to the LLM per iteration |

### `runner`

| Field | Default | Description |
|-------|---------|-------------|
| `type` | `docker` | One of `docker`, `k8s`, or `local` |
| `image` | `orchestra-musician:latest` | Docker image to use for training runs |
| `gpu_device` | `0` | GPU device index, `all`, or `none` |
| `timeout_seconds` | 3600 | Maximum wall-clock time per training run |
| `namespace` | `orchestra` | Kubernetes namespace (k8s runner only) |
| `node_selector` | `{}` | Kubernetes node selector labels (k8s runner only) |
| `k8s_job_ttl_seconds` | 300 | TTL for completed Kubernetes Jobs (k8s runner only) |

### `memory`

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable or disable semantic memory |
| `embedding_model` | `nomic-embed-text` | Ollama model used for embeddings |
| `embedding_url` | `http://localhost:11434` | Ollama base URL |
| `top_k` | 5 | Number of memories retrieved per iteration |
| `similarity_threshold` | 0.75 | Minimum cosine similarity for memory retrieval |

### `supabase`

| Field | Default | Description |
|-------|---------|-------------|
| `url_env` | `SUPABASE_URL` | Environment variable name for the project URL |
| `key_env` | `SUPABASE_ANON_KEY` | Environment variable name for the anon key |

### `program`

| Field | Default | Description |
|-------|---------|-------------|
| `path` | `program.md` | Path to the program specification (relative to `workspace_dir`) |
| `train_script` | `train.py` | Training script invoked per iteration |
| `eval_script` | `evaluate.py` | Evaluation script (optional) |
| `results_file` | `results.json` | JSON file written by the training script with metric results |
| `datasets_dir` | `~/.orchestra/datasets` | Local directory where datasets are stored |

---

## CLI Command Reference

All commands accept `--env /path/to/.env` to load secrets from a custom location.

```
orchestra init    --config FILE          Initialise a new session in Supabase
orchestra run     --config FILE          Start (or resume) a Conductor session
orchestra status  --config FILE [flags]  Show session status, memories, git log
orchestra migrate --config FILE          Apply database migrations
orchestra inspect --config FILE [flags]  Inspect memories, fallback history, best run
orchestra reset   --config FILE -n N     Revert workspace to iteration N
```

The launcher scripts (`orchestra.sh`, `orchestra.bat`, `orchestra.ps1`) also expose two additional commands:

```
setup    Run the interactive setup wizard (deploy/setup.py)
check    Run the health-check validator (deploy/check.py)
```

---

## Health Check

Run the validator at any time to diagnose configuration problems:

```bash
# Linux / macOS
python deploy/check.py

# Windows
python deploy\check.py

# Via launcher
./deploy/orchestra.sh check
.\deploy\orchestra.ps1 check
```

The validator checks:

- Python ≥ 3.11, Git, Docker, Ollama, kubectl
- `orchestra_sdk` package installation
- All required and optional environment variables
- Supabase REST API reachability and table existence
- LLM provider API key validity
- Ollama model availability
- Docker daemon status and `orchestra-musician:latest` image presence
- `conductor_config.yaml` field completeness

Exit code `0` means all critical checks passed. Exit code `1` means one or more critical checks failed.

---

## Building the Musician Docker Image

The Musician container is the training environment launched by the Conductor on each iteration. Build it once before starting a session with `runner.type: docker`.

```bash
# Linux / macOS — NVIDIA CUDA image (requires NVIDIA drivers)
./deploy/build_musician.sh

# Linux / macOS — CPU-only image (for testing without a GPU)
./deploy/build_musician.sh --cpu

# Windows CMD
deploy\build_musician.bat
deploy\build_musician.bat --cpu
```

The CUDA image is based on `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` and includes PyTorch 2.3.1, Unsloth, Transformers, PEFT, TRL, and all standard fine-tuning dependencies.

---

## Database Migrations

Migrations create and update all Supabase tables (`conductor_sessions`, `conductor_experiments`, `conductor_memories`, `session_best_runs`) and their RLS policies. They must be run once on a fresh project and again whenever the SDK is updated.

```bash
# Linux / macOS
./deploy/migrate.sh --config conductor_config.yaml

# Windows
deploy\migrate.bat --config conductor_config.yaml

# Dry-run (print SQL without executing)
./deploy/migrate.sh --config conductor_config.yaml --dry-run
```

Migrations require `SUPABASE_SERVICE_ROLE_KEY` to be set. If the `exec_sql` RPC function is not yet present in the project, run the first migration manually in the [Supabase SQL Editor](https://supabase.com/dashboard/project/domrhrldlufshogewfbp/sql/new).

---

## Rollback Procedures

If a session produces a degraded model, revert the workspace to any previous iteration:

```bash
# Find the iteration number from the dashboard or:
orchestra inspect --config conductor_config.yaml --git-log

# Revert to iteration 12
orchestra reset --config conductor_config.yaml --to-iteration 12

# Resume from the reverted state
orchestra run --config conductor_config.yaml --resume
```

The `reset` command performs a hard git reset of the workspace to the SHA recorded for the target iteration and updates the Supabase session record accordingly.

---

## Troubleshooting

**`supabaseKey is required` on the dashboard** — Add `VITE_SUPABASE_ANON_KEY` to the dashboard secrets or ensure the fallback in `client/src/lib/supabase.ts` is populated.

**DELETE operations silently do nothing** — The `datasets` and `dataset_samples` tables are missing permissive RLS DELETE policies. Run the SQL block in `deploy/README.md` or execute `orchestra migrate`.

**`orchestra` command not found after setup** — Activate the virtual environment: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows).

**Docker runner fails with `DeviceRequest` error on AMD GPU** — The AMD GPU path patches `DeviceRequest` with `/dev/kfd`. Ensure the ROCm drivers are installed and `/dev/kfd` is accessible.

**Kubernetes runner fails with `ApiException`** — Transient API errors are retried automatically. If the error persists, check that `kubectl` is configured for the correct cluster and the `orchestra` namespace exists.

**Memory embeddings not working** — Ensure Ollama is running (`ollama serve`) and the `nomic-embed-text` model is pulled (`ollama pull nomic-embed-text`). Set `memory.enabled: true` in `conductor_config.yaml`.
