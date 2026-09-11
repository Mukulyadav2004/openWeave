# openWeave

[![CI](https://github.com/Mukulyadav2004/openWeave/actions/workflows/ci.yml/badge.svg)](https://github.com/Mukulyadav2004/openWeave/actions/workflows/ci.yml)

An LLM observability and evaluation platform. Instrument an agent with a
decorator, get a nested trace with per-step token cost, have an LLM judge score
it automatically, and re-run a fixed dataset to find out whether your last
prompt change made things better or worse.

<!-- TODO: take a screenshot of your own data and uncomment this line. See
     "Screenshots" at the bottom.
![Trace waterfall](agent-observability/docs/screenshot-trace.png)
-->

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
| `tests/` | 44 tests, most of them regressions for bugs found in real runs |

## Quickstart

The whole stack, from the same image the deployment runs:

```bash
cd agent-observability
docker compose up -d --build     # Postgres, Redis, ingestion, worker, eval worker, API
docker compose exec api python scripts/bootstrap.py demo   # prints your API keys — once
docker compose exec api python scripts/seed_models.py      # model price table
docker compose exec api python scripts/seed_evaluator.py demo \
    --provider gemini --model gemini-3.1-flash-lite        # without this nothing is scored
```

The UI is at <http://localhost:8000>, served by the API itself. Scoring needs a
judge: export `GEMINI_API_KEY` before `docker compose up`, or seed the evaluator
with `--provider ollama --model llama3.2` and run Ollama yourself.

To run the services from a virtualenv instead, which is the loop you want while
developing, start only the databases:

```bash
docker compose up -d postgres redis
pip install -r requirements.txt
bash proto/generate.sh
export DATABASE_URL=postgresql+asyncpg://agentobs:agentobs@localhost:5432/agentobs
cd worker && alembic upgrade head && cd ..

python ingestion-server/server.py &
python worker/worker.py &
python evaluator/eval_worker.py &
uvicorn main:app --app-dir api-server --port 8000 &
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

The UI is served by the API itself at <http://localhost:8000>, so there is no
second server and no CORS to configure. A trace deep-links as
`/#trace=<trace-id>`.

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

44 tests. Most are regressions for bugs found running the thing end to end —
span closes rejected by a CHECK constraint, stub traces overwriting real ones,
an upsert clobbering fields the event never sent, a cached API key skipping its
expiry check, a debounce that a burst of announcements walked straight through,
an aggregate over a fan-out join double-counting cost, a price cache that
skipped its first load when the process started near boot. Each one fails on the
code as it was and passes on the code as it is.

## Deploy

`agent-observability/Dockerfile` builds one image; each service runs a different
command from it (API, ingestion, worker, eval worker), alongside Postgres and
Redis. The services read `DATABASE_URL` and `REDIS_URL` (password included), the
API binds `0.0.0.0` on `$PORT` and applies the migration on start, and
`PUBLIC_DEMO_PROJECT=<project>` lets anyone read that one project without a key
while every write still needs one. The ingestion server speaks gRPC rather than
HTTP, so on Railway it needs a TCP proxy rather than a domain.

<!-- TODO: paste the live demo URL here once it is deployed. -->

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

_Apple M5 (10 cores, 16 GB), macOS 25.6 (arm64), Python 3.12.13, 2026-09-11.
Postgres 15 and Redis 7 in Docker, every service on localhost — this measures the
code, not production capacity. One of three consecutive runs, chosen because it
sits at the median on nearly every metric. Reproduce with
`bash agent-observability/benchmark/run.sh`._

**SDK overhead per span** — what instrumentation costs the caller

| | p50 | p95 |
|---|---|---|
| Plain call | 0.167 µs | 0.209 µs |
| Instrumented | 15.208 µs | 18.0 µs |
| **Added by the SDK** | **15.041 µs** | 17.792 µs |

4601 events delivered, 0 dropped.

**Ingest throughput by batch size** — batch size 1 is v1's shape

| Batch size | RPCs | Events/sec |
|---|---|---|
| 1 | 5001 | 1,239 |
| 25 | 201 | 19,288 |
| 200 | 26 | 42,031 |

**End-to-end latency** (SDK call → readable in Postgres, on a drained stream, n=50): p50 12.921 ms · p95 15.886 ms · p99 16.607 ms

**Burst drain** — accept rate vs commit rate

| Burst | Accepted/sec | Committed in | Committed/sec |
|---|---|---|---|
| 5,001 GENERATION events | 43,441 | 1.283 s | 3,897 |
| 5,001 SPAN events | 43,436 | 0.329 s | 15,206 |

**API read latency**

| Endpoint | p50 | p95 |
|---|---|---|
| `GET /traces` (50 rows) | 3.853 ms | 4.285 ms |
| `GET /traces/{id}` (5,000-span tree) | 104.465 ms | 112.983 ms |

**API key cache**: p50 3.681 ms cold vs 1.941 ms warm — 1.74 ms saved per authenticated request.

**Reading these numbers**

- Batching is worth **34x** here (1,239 → 42,031 events/sec). v1 sent one blocking RPC per trace, so the batch-size-1 row is roughly its ceiling.
- The SDK adds **~15 µs** to a decorated call — building the span's create and update events (ids, timestamps, protobuf) and an in-memory enqueue; nothing waits on the network. v1 held the caller for a full gRPC round trip instead. Argument capture is a small share for small arguments but grows with their size, so pass `capture_input=False` on functions that take large inputs.
- End-to-end p50 **13 ms**, p95 16 ms, measured once the previous stage's backlog has drained. Without that wait the first samples queue behind it, and at n=50 a single slow sample becomes the p99. A blocked XREADGROUP returns as soon as an event arrives, so an idle worker adds no poll delay.
- Ingest accepts **43,441/sec**, but the worker commits generations at **3,897/sec** and plain spans at 15,206/sec. The gap is cost resolution, which issues one UPDATE per priced generation, so that is the first thing to batch if commit rate matters. Accepting faster than you commit only moves the queue: the commit rate is what the stream's MAXLEN and the worker replica count have to be sized against.
- Throughput and drain vary between runs, and the first run against a fresh database is noticeably slower. Discard it and report the median of the next few.

## Screenshots

Regenerate these from your own data before committing them:

1. Start the stack and the UI, run some real traffic through it.
2. Screenshot the trace view with a trace that has a nested tool call.
3. Save to `agent-observability/docs/screenshot-trace.png`.

---

Built with Python, gRPC, Redis Streams, PostgreSQL, SQLAlchemy and FastAPI.
