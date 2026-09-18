# openWeave

[![CI](https://github.com/Mukulyadav2004/openWeave/actions/workflows/ci.yml/badge.svg)](https://github.com/Mukulyadav2004/openWeave/actions/workflows/ci.yml)

A small LLM observability and evaluation platform inspired by Langfuse. It
captures agent traces, nested spans, token usage, cost, latency, and evaluation
scores in one dashboard.

![Trace waterfall](agent-observability/docs/screenshot-trace.png)

## Features

- Python SDK with an `@observe` decorator
- Nested trace and span waterfall
- Token usage, cost, and latency tracking
- LLM-as-a-judge evaluation with Gemini or Ollama
- Dataset experiments and prompt-version comparison
- Dashboard for traffic, cost, latency, and scores
- PostgreSQL storage and Redis-based event processing

## Run locally

Requirements: Docker and Docker Compose.

```bash
git clone https://github.com/Mukulyadav2004/openWeave.git
cd openWeave/agent-observability
docker compose up -d --build
```

Create a project and API keys:

```bash
docker compose exec api python scripts/bootstrap.py demo
docker compose exec api python scripts/seed_models.py
```

Save the `public key` and `secret key` printed by the bootstrap command. Open
<http://localhost:8000> and use those keys to connect.

## Enable Gemini evaluations

Create `agent-observability/.env` and place your real key after the `=` sign:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Then restart the evaluation worker and configure the evaluator:

```bash
docker compose up -d --force-recreate eval-worker
docker compose exec api python scripts/seed_evaluator.py demo \
  --provider gemini --model gemini-3.1-flash-lite
```

The `.env` file is ignored by Git and must not be committed.

## Run the demo experiment

Run the demo inside the API container. Replace the two placeholder values with
the keys printed by `bootstrap.py`:

```bash
docker compose exec \
  -e OPENWEAVE_PUBLIC_KEY=paste_the_public_key_here \
  -e OPENWEAVE_SECRET_KEY=paste_the_secret_key_here \
  -e OPENWEAVE_HOST=ingestion \
  api python scripts/demo_experiment.py
```

The experiment compares an incorrect, ungrounded baseline (`prompt-v1`) with a
grounded answer (`prompt-v2`). After it finishes, open the Experiments tab to
compare quality and cost.

## Instrument Python code

```python
from openweave import OpenWeave

ow = OpenWeave()

@ow.observe(type="RETRIEVER")
def search(question):
    return find_documents(question)

@ow.observe(type="AGENT")
def answer(question):
    return generate_answer(question, search(question))
```

The SDK reads `OPENWEAVE_PUBLIC_KEY` and `OPENWEAVE_SECRET_KEY` from the
environment. Nested decorated calls appear as nested spans in the trace view.

## Tests

Use a dedicated test database because the test suite resets its schema:

```bash
cd agent-observability
docker compose exec postgres createdb -U agentobs agentobs_test
TEST_DATABASE_URL=postgresql+asyncpg://agentobs:agentobs@localhost:5432/agentobs_test \
  pytest tests/ -q
```

## Project structure

| Directory | Purpose |
|---|---|
| `sdk/` | Python tracing SDK and experiment runner |
| `ingestion-server/` | Receives batched trace events over gRPC |
| `worker/` | Stores events and calculates cost |
| `evaluator/` | Runs automatic LLM evaluations |
| `api-server/` | FastAPI read and experiment API |
| `ui/` | Dashboard and trace explorer |
| `tests/` | Integration and regression tests |

More implementation details are in
[`agent-observability/docs/M1-DESIGN-NOTES.md`](agent-observability/docs/M1-DESIGN-NOTES.md).
