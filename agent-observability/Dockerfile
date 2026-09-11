# One image for every service. Each service picks its process with a start
# command (docker-compose.yml and the Railway services use exactly these):
#
#   api          (the default CMD below)
#   ingestion    python ingestion-server/server.py
#   worker       python worker/worker.py
#   eval-worker  python evaluator/eval_worker.py
#
# Both servers bind 0.0.0.0 by default, which every container runtime can reach.
# On a host that routes to containers over IPv6 (Railway's private network), set
# UVICORN_HOST=:: for the api service and GRPC_BIND=:: for the ingestion one.
#
# Built from agent-observability/ so every service can import shared/.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN bash proto/generate.sh

EXPOSE 8000 50051
CMD ["sh", "-c", "cd worker && alembic upgrade head && cd .. && exec uvicorn main:app --app-dir api-server --host ${UVICORN_HOST:-0.0.0.0} --port ${PORT:-8000}"]
