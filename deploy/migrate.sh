#!/usr/bin/env bash
# Orchestra SDK — Run database migrations
# =========================================
# Convenience wrapper that loads .env and runs `orchestra migrate`.
# Requires SUPABASE_SERVICE_ROLE_KEY to be set for DDL operations.
#
# Usage:
#   chmod +x deploy/migrate.sh
#   ./deploy/migrate.sh --config conductor_config.yaml
#   ./deploy/migrate.sh --config conductor_config.yaml --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"
ENV_FILE="$REPO_ROOT/.env"

if [[ ! -d "$VENV" ]]; then
    echo "✗  Virtual environment not found. Run: python deploy/setup.py"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Require service role key for migrations
if [[ -z "${SUPABASE_SERVICE_ROLE_KEY:-}" ]]; then
    echo "⚠  SUPABASE_SERVICE_ROLE_KEY is not set."
    echo "   Migrations require the service role key to execute DDL."
    echo "   Add it to .env or set it in the environment."
    echo "   Find it at: https://supabase.com/dashboard/project/_/settings/api"
    exit 1
fi

# Export as the key the SDK expects
export SUPABASE_SERVICE_ROLE_KEY

echo "Running Orchestra migrations …"
orchestra migrate --env "$ENV_FILE" "$@"
