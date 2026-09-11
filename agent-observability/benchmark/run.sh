#!/usr/bin/env bash
# Runs benchmark/load_test.py against a local stack. Arguments pass through:
#
#   bash benchmark/run.sh --quick
#   bash benchmark/run.sh --json bench.json
#
# Use keys for a project with no evaluation rules (python scripts/bootstrap.py
# bench), so the eval worker does not send benchmark traffic to the judge mid-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -c "import grpc, asyncpg, redis, httpx" 2>/dev/null || {
  echo "Missing dependencies: pip install -r requirements.txt"
  exit 1
}
if [[ -z "${OPENWEAVE_PUBLIC_KEY:-}" || -z "${OPENWEAVE_SECRET_KEY:-}" ]]; then
  echo "Set OPENWEAVE_PUBLIC_KEY and OPENWEAVE_SECRET_KEY (python scripts/bootstrap.py bench prints them)."
  exit 1
fi

export OPENWEAVE_HOST="${OPENWEAVE_HOST:-localhost}"
export OPENWEAVE_PORT="${OPENWEAVE_PORT:-50051}"
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export DATABASE_URL="${DATABASE_URL:-postgresql://agentobs:agentobs@localhost:5432/agentobs}"
export API_BASE="${API_BASE:-http://localhost:8000}"

exec python3 "${SCRIPT_DIR}/load_test.py" "$@"
