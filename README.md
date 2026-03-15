# orchestra_sdk

Autonomous LLM-driven research loop for iterative LLM fine-tuning.

## Quick Start

```bash
pip install orchestra-sdk
orchestra init my_session
orchestra migrate
orchestra run conductor_config.yaml
```

## Architecture

The Conductor runs a 10-step loop per iteration:
1. Read `program.md` (research goals and constraints)
2. Search memories (semantic retrieval via pgvector)
3. Read git log (recent changes)
4. Read metric history from Supabase
5. Propose hypothesis (LLM structured output → `Hypothesis` schema)
6. Apply edit to `train.py` (find/replace)
7. Commit candidate to git
8. Run experiment (Docker or K8s)
9. Read `results.json`
10. Keep or discard (based on `keep_threshold`)

## Installation

```bash
pip install orchestra-sdk
# or for development:
git clone https://github.com/your-org/orchestra-sdk
cd orchestra-sdk
pip install -e ".[dev]"
```

## Configuration

See `examples/memory_scribe/conductor_config.yaml` for a complete example.

## Running Tests

```bash
pytest tests/ -v
```
