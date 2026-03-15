# Orchestra SDK — Quick Start Guide

## Prerequisites

- Python 3.11+
- Docker with NVIDIA GPU support (for training)
- An LLM API key (OpenRouter recommended)
- Supabase project with the Orchestra schema applied

---

## 1. Install

```bash
git clone https://github.com/your-org/orchestra-sdk
cd orchestra-sdk
pip install -e .
```

---

## 2. Apply Database Migrations

Run the migration to create the three Conductor tables in Supabase:

```bash
orchestra migrate --supabase-url $SUPABASE_URL --supabase-key $SUPABASE_ANON_KEY
```

This creates:
- `conductor_sessions` — one row per session, tracks baseline metric and iteration count
- `conductor_experiments` — one row per iteration, stores hypothesis, metric, decision
- `conductor_memories` — vector embeddings for semantic memory retrieval

---

## 3. Prepare Your Workspace

```bash
mkdir -p ~/.orchestra/workspace/memory_scribe_v1
cd ~/.orchestra/workspace/memory_scribe_v1

# Copy the example files
cp /path/to/orchestra-sdk/examples/memory_scribe/train.py .
cp /path/to/orchestra-sdk/examples/memory_scribe/program.md .
cp /path/to/orchestra-sdk/examples/memory_scribe/conductor_config.yaml .
```

Edit `conductor_config.yaml` and set:
- `session.workspace_dir` — absolute path to your workspace
- `llm.api_key_env` — name of your API key env var
- `runner.gpu_device` — your GPU device index

---

## 4. Set Environment Variables

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_ANON_KEY=eyJ...
```

---

## 5. Validate Configuration

```bash
orchestra dry-run conductor_config.yaml
```

This validates config, checks workspace files, tests LLM connectivity, and verifies Supabase access — without running any experiments.

---

## 6. Run the Conductor

```bash
orchestra run conductor_config.yaml
```

The Conductor will:
1. Initialize the session and commit the initial workspace to git
2. For each iteration (up to `max_iterations`):
   - Read `program.md` and recent git history
   - Search semantic memories for relevant past experiments
   - Propose a hypothesis (LLM structured output)
   - Apply the edit to `train.py`
   - Commit the candidate
   - Launch the Musician container (`docker run orchestra-musician`)
   - Read `results.json`
   - Keep or discard based on `keep_threshold`
   - Log to Supabase and update memory
3. Stop when `target_value` is reached or `max_iterations` exhausted

---

## 7. Monitor Progress

While the Conductor runs, you can monitor in the Orchestra Dashboard:
- **Conductors** tab → view session status, iteration count, baseline metric
- **Metrics** tab → view per-iteration metric history and keep/discard decisions

---

## 8. Resume a Session

If the Conductor is interrupted, resume from the last committed state:

```bash
orchestra run conductor_config.yaml --resume
```

---

## Building the Musician Image

```bash
docker build -f docker/Dockerfile.musician -t orchestra-musician:latest .
```

For AMD ROCm GPUs:
```bash
docker build -f docker/Dockerfile.musician.rocm -t orchestra-musician:rocm .
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 32 tests should pass. The integration test runs a full 5-iteration loop with mocked LLM and runner, validating keep/discard logic end-to-end.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: Environment variable 'OPENROUTER_API_KEY' is not set` | Missing env var | `export OPENROUTER_API_KEY=...` |
| `EditFileError: find string not found` | LLM proposed a find string that doesn't match | Check `train.py` — the LLM may have hallucinated a line. Edit `train.py` to match or add the expected line. |
| `RunExperimentError: Container OOM` | GPU VRAM exceeded | Reduce `BATCH_SIZE` or `LORA_RANK` in `train.py`, or set `GRADIENT_ACCUMULATION_STEPS` higher |
| `ResultsNotFoundError` | `train.py` didn't write `results.json` | Check container logs: `docker logs orchestra-musician-<id>` |
| `StructuredOutputError: max retries exceeded` | LLM returned invalid JSON | Try a more capable model (e.g., `claude-3-7-sonnet`) or increase `max_tokens` |
| Memory search returns no results | Ollama not running or model not pulled | `ollama pull nomic-embed-text && ollama serve` |
