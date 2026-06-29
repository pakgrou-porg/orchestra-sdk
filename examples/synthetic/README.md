# Synthetic Test Example

This example runs the full Orchestra Conductor loop — LLM proposal, git commit, training run, metric evaluation, keep/discard decision, and Supabase logging — **without a GPU, model weights, or a real dataset**.

It is the recommended starting point for:
- Validating a new SDK installation
- Testing Conductor loop changes in CI
- Demonstrating the framework to new team members
- Manus-driven automated testing

---

## Prerequisites

```bash
pip install -e /path/to/orchestra-sdk

export SUPABASE_URL=https://<your-project>.supabase.co
# Trusted backend key, used by the Conductor at runtime and for migrations:
export SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
export OPENROUTER_API_KEY=<your-key>
```

Run migrations once (creates all required tables):

```bash
orchestra migrate --config examples/synthetic/conductor_config.yaml
```

---

## Option A — Local runner (no Docker)

The `local` runner type executes `train.py` directly in the host Python process. No Docker installation required.

```bash
orchestra run --config examples/synthetic/conductor_config.yaml
```

---

## Option B — Docker runner (no GPU)

Build the minimal Musician image (uses `python:3.11-slim`, no CUDA):

```bash
docker build -f examples/synthetic/Dockerfile -t orchestra-musician-synthetic .
```

Update `conductor_config.yaml`:

```yaml
runner:
  type: "docker"
  image: "orchestra-musician-synthetic:latest"
  gpu_device: "none"
```

Then run:

```bash
orchestra run --config examples/synthetic/conductor_config.yaml
```

---

## What the Conductor will do

1. Read `train.py` and `program.md`
2. Propose a change to 1–3 hyperparameters with a written hypothesis
3. Commit the change to git
4. Run `train.py` (completes in ~3 seconds)
5. Read `results.json` and compare `val_loss` to the previous best
6. Keep or discard the change
7. Write the experiment to Supabase and (if memory is enabled) store a vector memory
8. Repeat up to 20 iterations, targeting `val_loss ≤ 2.30`

---

## Simulating a crash

To test `FAILED` handling, set `SIMULATE_CRASH=1` in the runner environment:

```bash
SIMULATE_CRASH=1 python train.py
```

The script will raise a `RuntimeError` with an OOM-style message, which the Conductor will handle as a FAILED iteration and revert the workspace.

---

## Inspecting the session

```bash
orchestra inspect --config examples/synthetic/conductor_config.yaml --all
```

Reverting to a specific iteration:

```bash
orchestra reset --config examples/synthetic/conductor_config.yaml --to-iteration 5
```
