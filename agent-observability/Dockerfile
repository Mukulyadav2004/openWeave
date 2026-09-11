# One image for every service. Each service sets SERVICE_ROLE to pick its
# process — api (default), ingestion, worker, eval-worker, or worker+eval for
# hosts that limit how many services a project may run. See scripts/start.sh.
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
RUN bash proto/generate.sh && chmod +x scripts/start.sh

EXPOSE 8000 50051
CMD ["bash", "scripts/start.sh"]
