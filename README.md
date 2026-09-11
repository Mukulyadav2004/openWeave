# openWeave

[![CI](https://github.com/Mukulyadav2004/openWeave/actions/workflows/ci.yml/badge.svg)](https://github.com/Mukulyadav2004/openWeave/actions/workflows/ci.yml)

An LLM observability and evaluation platform. Instrument an agent with a
decorator, get a nested trace with per-step token cost, have an LLM judge score
it automatically, and re-run a fixed dataset to find out whether your last
prompt change made things better or worse.

<!-- TODO: replace with your own screenshot. See "Screenshots" at the bottom. -->
![Trace waterfall](agent-observability/docs/screenshot-trace.png)

---

## Why it exists

Tracing tells you what your agent did. It does not tell you whether it did it
*well*, and it does not tell you whether your last change helped. openWeave
closes that loop:

```
change a prompt → re-run a fixed dataset → compare scores and cost across runs
```

Everything else in the system exists to make that loop trustworthy — nested
spans so a score can be attributed to a step, versioned judges so a score drop
is unambiguous, per-observation cost so "better" can be weighed against "more
expensive".

## Architecture

```
   your app                 @ow.observe decorator; contextvars span stack;
      │                     background batched flush — never blocks the caller,
      │  gRPC (batched)     never changes what your code returns or raises
      ▼
 ┌──────────────────┐       HTTP Basic → sha256 → Redis cache → Postgres.
 │ ingestion server │       project_id comes from the key, never the client.
 └────────┬─────────┘
          │  XADD (MAXLEN-bounded)
          ▼
   ┌─────────────┐          Redis Stream, consumer group
   │   events    │
   └──────┬──────┘
          │  XREADGROUP + XAUTOCLAIM
          ▼
 ┌──────────────────┐  ──►  PostgreSQL   (traces, observations, scores, …)
 │      worker      │  ──►  dead letters (replayable, not discarded)
 └────────┬─────────┘
          │  cost engine: regex model match, price as-of the observation
          │  ZADD eval:due  (debounce — wait until the trace goes quiet)
          ▼
 ┌──────────────────┐  ──►  judge: Ollama or any OpenAI-compatible endpoint
 │   eval worker    │       versioned prompt, schema-validated verdict
 └────────┬─────────┘  ──►  scores + job_executions (successes AND failures)
          │
          ▼
 ┌──────────────────┐  ◄──  single-file UI: waterfall, costs, run comparison
 │   FastAPI API    │
 └──────────────────┘
```

Everything lives under `agent-observability/`:

| Path | What it is |
|---|---|
| `proto/` | Protobuf schema (batched event envelope) + stub generation |
| `sdk/` | Python SDK and the dataset/experiment runner |
| `ingestion-server/` | Authenticated gRPC ingest → Redis Stream |
| `worker/` | Stream consumer, upserts, cost resolution; `models.py` is the schema |
| `evaluator/` | Eval worker + judge adapters |
| `api-server/` | Read API, datasets, run comparison |
| `ui/` | Single-file trace explorer (no build step) |
| `shared/` | Auth and pricing, used by more than one service |
| `tests/` | 39 tests, most of them regressions for bugs found in real runs |

## Quickstart

```bash
cd agent-observability
pip install -r requirements.txt
bash proto/generate.sh
docker compose up -d postgres redis ollama   # leave out ollama if you run it natively
export DATABASE_URL=postgresql+asyncpg://agentobs:agentobs@localhost:5432/agentobs

cd worker && alembic upgrade head && cd ..

python scripts/bootstrap.py demo        # prints your API keys — once
python scripts/seed_models.py           # model price table
python scripts/seed_evaluator.py demo   # the judge; without this nothing is scored

python ingestion-server/server.py &
python worker/worker.py &
python evaluator/eval_worker.py &
CORS_ORIGINS=http://localhost:8080 uvicorn main:app --app-dir api-server --port 8000 &
```

Instrument something:

```python
from openweave import OpenWeave

ow = OpenWeave()   # reads OPENWEAVE_PUBLIC_KEY / OPENWEAVE_SECRET_KEY

@ow.observe(type="RETRIEVER")
def search(question): ...

@ow.observe(type="GENERATION")
def answer(question, docs):
    r = client.chat.completions.create(...)
    ow.current().update(
        model=r.model,
        usage={"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens},
    )
    return r.choices[0].message.content

@ow.observe(type="AGENT")
def handle(question):
    return answer(question, search(question))   # nests automatically
```

Open the UI:

```bash
cd ui && python -m http.server 8080
```

Then open <http://localhost:8080> — `localhost` exactly, since the API only
allows the origins listed in `CORS_ORIGINS` and `127.0.0.1` is a different one.

## The experiment loop

```python
from experiment import Experiment

exp = Experiment(ow, api_base="http://localhost:8000")
exp.create_dataset("support-qa")
exp.add_item("support-qa", "How do I reset my password?", "Use the reset link.")

exp.run("support-qa", "prompt-v1", agent_v1, metadata={"prompt_version": 1})
exp.run("support-qa", "prompt-v2", agent_v2, metadata={"prompt_version": 2})

print(exp.format_comparison(exp.compare("support-qa", ["prompt-v1", "prompt-v2"])))
```

<!-- TODO: paste YOUR OWN comparison output here, from a real judge and a real
     agent. Do not ship numbers from the simulated demo agent — they are not a
     result and an interviewer who asks how they were produced will find that
     out in one question. `scripts/demo_experiment.py` is a wiring check, not a
     benchmark. -->

Scores reach a run through `dataset_run_items`, so the same judge and the same
score rows serve production traffic and offline experiments — there is no
separate experiment-scoring path to keep in sync.

## Design decisions worth reading

Full rationale lives in [`docs/M1-DESIGN-NOTES.md`](agent-observability/docs/M1-DESIGN-NOTES.md).
The short version:

- **Composite primary keys `(project_id, id)`** on every tenant-owned table.
  The classic multi-tenant bug is one endpoint that forgets `WHERE project_id =`;
  with a composite key that endpoint does not typecheck.
- **No foreign key from `observations` to `traces`.** Events arrive over a queue
  and are reordered; a child span routinely lands before its parent trace. The
  worker upserts a stub instead. Referential integrity is enforced at read time
  because the transport does not guarantee ordering — and the read path returns
  orphaned spans flagged rather than dropping them.
- **Usage and cost are JSONB maps, not columns.** `cache_read` and `reasoning`
  tokens both appeared after most schemas were written.
- **Prices carry a `start_date`.** Re-costing a trace from three months ago uses
  the price that applied *then*, so a vendor price change cannot silently
  rewrite historical spend.
- **Scores carry a `source` (`API` / `EVAL` / `ANNOTATION`) and an evaluator
  version.** Judge output and human labels live in one table with one shape, so
  judge-vs-human agreement is a single query; and a score drop is never
  ambiguous between "the agent got worse" and "someone edited the judge".
- **The eval debounce is a Redis sorted set, not a re-queue.** Time-based state
  belongs in a structure keyed by time. See the note in `evaluator/eval_worker.py`
  for what happens when you try to do it with a bounce counter instead.
- **Sampling is a hash of the trace id, not `random()`.** A trace is announced
  many times, so `random()` would let it re-roll past a 5% rate.

## Tests

```bash
cd agent-observability
docker compose exec postgres createdb -U agentobs agentobs_test
TEST_DATABASE_URL=postgresql+asyncpg://agentobs:agentobs@localhost:5432/agentobs_test pytest tests/ -q
```

> The suite calls `drop_all`. Point it at a **dedicated** test database.

39 tests. Most are regressions for bugs found running the thing end to end —
span closes rejected by a CHECK constraint, stub traces overwriting real ones,
an upsert clobbering fields the event never sent, a cached API key skipping its
expiry check, a debounce that a burst of announcements walked straight through,
an aggregate over a fan-out join double-counting cost, a price cache that
skipped its first load when the process started near boot. Each one fails on the
code as it was and passes on the code as it is.

## Not built, on purpose

| | Why |
|---|---|
| OTLP ingest | Would make the ecosystem's instrumentation work out of the box. The clear next thing; the wire format here is bespoke until then. |
| ClickHouse | A scale answer, and scale is not this project's constraint. The schema *shape* is what matters and Postgres holds it fine at this size. |
| Organizations above projects | One tenancy level proves the pattern and costs no RBAC matrix. |
| Dataset item versioning | Correct, but adds a dimension to every dataset query. |
| Prompt registry | `prompt_name` / `prompt_version` already ride on every observation, so grouping runs by prompt version works without the table. |
| SSO, billing, entitlements | Business features, not engineering. |

## Benchmarks

`benchmark/load_test.py` still speaks the v1 protocol and has to be ported to
the v2 SDK before it can measure ingestion throughput and end-to-end latency.

<!-- TODO: port it, run it, and paste the numbers, with the hardware and configuration
     they came from. Then re-run after any change that should move them and show
     both. "I measured it, changed the design, measured again" is worth more
     than any single number — and an unlabelled number is worth nothing. -->

## Screenshots

Regenerate these from your own data before committing them:

1. Start the stack and the UI, run some real traffic through it.
2. Screenshot the trace view with a trace that has a nested tool call.
3. Save to `agent-observability/docs/screenshot-trace.png`.

---

Built with Python, gRPC, Redis Streams, PostgreSQL, SQLAlchemy and FastAPI. The
`k8s/` manifests and `spark-jobs/` predate the v2 rewrite and have not been
updated for it.
