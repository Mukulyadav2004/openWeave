# Agent Observability & Evaluation Platform

Backend for capturing and evaluating traces from AI agent apps. Your app uses the Python SDK to send traces to an ingestion server, which drops them onto a message queue. A worker picks them up and writes to a database. From there, an LLM-as-judge evaluator scores each trace, and a REST API lets you query everything. A nightly batch job runs anomaly detection on top.

## How it works

```
   Your AI app
       │
       │  sends traces
       ▼
   Ingestion Server
       │
       │  queues them
       ▼
   Message Queue
       │
       │  picked up by
       ▼
   Worker  ──────────────────▶  Database
                                    │
                          ┌─────────┴──────────┐
                          │                    │
                          ▼                    ▼
                     Evaluator            REST API
                  (scores each         (query traces
                   trace with          & evaluations)
                    an LLM)

   Anomaly Detector  ◀────────────────  Database
   (runs nightly)
```

## What's in here

| Path | What it does |
|---|---|
| `proto/` | Protobuf schema and stub generation script |
| `ingestion-server/` | gRPC server that receives traces |
| `sdk/` | Python SDK for instrumenting your app |
| `worker/` | Redis consumer that writes traces to Postgres |
| `evaluator/` | Lambda-style LLM-as-judge evaluator (Ollama) |
| `api-server/` | FastAPI server — query traces and evaluations |
| `spark-jobs/` | Nightly anomaly detection batch job |
| `k8s/` | Kubernetes manifests for minikube |
| `docker-compose.yml` | Spins up Postgres, Redis, Ollama locally |

## Running locally

```bash
pip install -r requirements.txt

# generate gRPC stubs
bash proto/generate.sh

docker compose up -d
```
