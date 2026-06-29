"""
orchestra_sdk.config_template
================================
Generates starter conductor_config.yaml files.
"""

from __future__ import annotations


def render_template(session_name: str) -> str:
    """Render a starter conductor_config.yaml for a new session."""
    return f"""\
# Orchestra Conductor Configuration
# Generated for session: {session_name}
# Edit all fields marked with <REQUIRED> before running.

session:
  name: "{session_name}"
  dataset_id: "{session_name}"          # Must match a dataset in your Supabase datasets table
  workspace_dir: "~/.orchestra/workspace/{session_name}"
  branch: "session/{session_name}"
  max_iterations: 50
  target_metric: "val_loss"             # Key in results.json to minimize
  target_value: null                    # Optional: stop when metric reaches this value
  keep_threshold: -0.005                # Keep commit if delta <= this (must be negative)

llm:
  provider: "openrouter"                # openrouter | anthropic | openai | lmstudio
  model: "anthropic/claude-3-7-sonnet-20250219"
  api_key_env: "OPENROUTER_API_KEY"     # Name of env var holding the API key
  max_tokens: 4096
  temperature: 0.3
  context_budget_tokens: 8000

runner:
  type: "docker"                        # docker | k8s
  image: "orchestra-musician:latest"    # <REQUIRED> Build this image first
  gpu_device: "0"                       # "0" | "all" | "none"
  timeout_seconds: 3600

  # K8s-specific (only used when type: k8s)
  # node_selector:
  #   orchestra/gpu: "rtx4060"
  # namespace: "orchestra"

memory:
  embedding_model: "nomic-embed-text"
  embedding_url: "http://localhost:11434"  # Ollama endpoint
  top_k: 5
  similarity_threshold: 0.75
  enabled: true

supabase:
  url_env: "SUPABASE_URL"
  # The Conductor is a trusted backend: use the service-role/secret key so the
  # database can enforce strict RLS (the public anon key stays read-only).
  # NEVER expose this key in any browser/client app.
  key_env: "SUPABASE_SERVICE_ROLE_KEY"
  anon_key_env: "SUPABASE_ANON_KEY"   # fallback only; not recommended for runtime
  session_table: "conductor_sessions"
  experiments_table: "conductor_experiments"
  memories_table: "conductor_memories"

program:
  path: "program.md"                    # Research program specification (relative to workspace_dir)
  train_script: "train.py"             # Script the Conductor edits (relative to workspace_dir)
  eval_script: "evaluate.py"           # Read-only evaluation harness
  results_file: "results.json"         # Written by train.py, read by Conductor
  datasets_dir: "~/.orchestra/datasets"
"""
