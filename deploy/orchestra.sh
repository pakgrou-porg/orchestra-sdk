#!/usr/bin/env bash
# Orchestra SDK — Linux / macOS launcher
# ========================================
# Activates the virtual environment, loads .env, and passes all arguments
# to the `orchestra` CLI.
#
# Usage:
#   chmod +x deploy/orchestra.sh
#   ./deploy/orchestra.sh run --config conductor_config.yaml
#   ./deploy/orchestra.sh status --config conductor_config.yaml
#   ./deploy/orchestra.sh migrate --config conductor_config.yaml
#   ./deploy/orchestra.sh check          # runs the health-check validator

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"
ENV_FILE="$REPO_ROOT/.env"

# ── Activate virtual environment ─────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "✗  Virtual environment not found at $VENV"
    echo "   Run: python deploy/setup.py"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── Load .env ────────────────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "⚠  .env not found at $ENV_FILE"
    echo "   Run: python deploy/setup.py  or  cp deploy/.env.example .env"
fi

# ── Dispatch ─────────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "Orchestra SDK launcher"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup                   Run the interactive setup wizard"
    echo "  check                   Run the health-check validator"
    echo "  run     --config FILE   Start a Conductor session"
    echo "  status  --config FILE   Show session status"
    echo "  migrate --config FILE   Apply database migrations"
    echo "  inspect --config FILE   Inspect session memories and git log"
    echo "  reset   --config FILE   Revert workspace to a previous iteration"
    echo ""
    echo "Examples:"
    echo "  $0 run --config conductor_config.yaml"
    echo "  $0 status --config conductor_config.yaml --all"
    exit 0
fi

COMMAND="${1}"
shift

case "$COMMAND" in
    setup)
        python "$REPO_ROOT/deploy/setup.py" "$@"
        ;;
    check)
        python "$REPO_ROOT/deploy/check.py" --env "$ENV_FILE" "$@"
        ;;
    *)
        orchestra "$COMMAND" --env "$ENV_FILE" "$@"
        ;;
esac
