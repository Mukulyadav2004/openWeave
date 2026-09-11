#!/usr/bin/env bash
# One image, several roles. The role comes from SERVICE_ROLE rather than a
# per-service start command, so every deployment runs the identical image and
# only one variable changes between them.
set -euo pipefail

role="${SERVICE_ROLE:-api}"

case "$role" in
  api)
    # The migration runs here rather than in a separate release step: it is
    # idempotent, and the API is the one service that must not serve traffic
    # against a schema that has not been applied yet.
    (cd worker && alembic upgrade head)
    exec uvicorn main:app --app-dir api-server \
         --host "${UVICORN_HOST:-0.0.0.0}" --port "${PORT:-8000}"
    ;;
  ingestion)
    exec python ingestion-server/server.py
    ;;
  worker)
    exec python worker/worker.py
    ;;
  eval-worker)
    exec python evaluator/eval_worker.py
    ;;
  worker+eval)
    # Both consumers in one container, for hosts that cap how many services a
    # project may have. If either exits, exit non-zero so the platform restarts
    # the container instead of quietly running half of it.
    python worker/worker.py &
    python evaluator/eval_worker.py &
    wait -n
    exit 1
    ;;
  *)
    echo "unknown SERVICE_ROLE: $role (api|ingestion|worker|eval-worker|worker+eval)" >&2
    exit 2
    ;;
esac
