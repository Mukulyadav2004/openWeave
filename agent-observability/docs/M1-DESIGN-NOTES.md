# M1 — Foundation: design notes

Scope: 3 weeks, AI/ML infrastructure roles, scaffold-and-implement.

This document covers what changed in the data model and why, what you build on
top of it, and what was deliberately left out. The "why" sections are written
so you can defend each decision in an interview — that is the actual point of
the exercise, not the code.

---

## The 3-week plan

Reordered from the original gap list because the target is AI infra: the eval
loop is pulled forward, and full reliability hardening is cut to three fixes.

### Week 1 — Foundation

| Day | Work | Gap |
|---|---|---|
| 1 | Review `db/models.py` and `proto/openweave.proto`, change what you disagree with | 1, 2, 5 |
| 2 | Alembic migration for the new schema; delete `init_db()`/`create_all()` | 1 |
| 3 | Regenerate stubs; API key auth — gRPC interceptor + FastAPI dependency | 2 |
| 4 | Worker rewrite: batch events, upsert on `event_id`, stub-trace creation | 1 |
| 5 | pytest + testcontainers harness, GitHub Actions CI green | 7 |

### Week 2 — SDK and the eval loop

| Day | Work | Gap |
|---|---|---|
| 6–7 | SDK v2: contextvars span stack, `@observe`, sync+async, background flush, error capture | 9 |
| 8 | Three reliability fixes: consumer name from hostname, `XAUTOCLAIM`, DLQ | 8 |
| 9 | Cost engine: regex model match, price lookup, `cost_details` resolution | 6 |
| 10 | Eval queue: worker publishes → eval consumer → versioned judge → `scores` row | 3 |

### Week 3 — Experiments and presentation

| Day | Work | Gap |
|---|---|---|
| 11–12 | Datasets, items, runs, run items; `add_to_dataset(trace_id)` | 4 |
| 13 | Experiment runner + run comparison endpoint | 4 |
| 14 | REST API v2: traces tree, observations, scores, datasets, runs, metrics | — |
| 15 | Test pass to a coverage number you can put in the README | 7 |
| 16–17 | README rewrite, architecture diagram, **publish your benchmark numbers** | — |
| 18–19 | Stretch, pick one: OTLP ingest endpoint, or a thin trace-waterfall UI | 10 / 11 |

**If Week 3 slips, ship what you have.** A tested, multi-tenant, nested-trace
platform with a working eval loop is a strong project. A half-built version of
all eleven gaps is not.

---

## What changed in the data model

### Flat trace → observation tree

v1's `Trace` had one input, one output, one latency, one token count. An agent
run is a tree: agent → tool → LLM → retriever → sub-agent.

`observations` is self-referencing via `parent_observation_id`, typed by
`ObservationType`, and carries `start_time` / `end_time` separately. That is
what makes a waterfall renderable and what makes per-step cost attribution
possible. Everything else in M1 depends on this table.

The old `SpanEvent` had a single `timestamp` and no parent — so it could record
that something happened, but not what it was part of or how long it took.

### Composite primary keys `(project_id, id)`

Every tenant-owned table uses a composite PK. This makes tenant isolation a
property of the schema instead of a thing you have to remember:

```python
# Cannot compile without a project_id. That is the point.
session.get(Trace, {"project_id": pid, "id": tid})
```

The classic multi-tenant bug is one endpoint that forgets `WHERE project_id =`.
With a composite PK, that endpoint does not typecheck.

### No FK from `observations` → `traces`

Deliberate. Events arrive over a Redis Stream and can be reordered; a child
span routinely lands before its parent trace event. A hard FK would either
reject valid data or force you to serialise the stream (destroying throughput).

The worker upserts a stub trace row when it sees an observation for an unknown
trace, and the real `trace_create` event later fills in the details. This is
the standard trade in event-driven ingestion and it is a good interview answer:
*referential integrity is enforced at read time, not write time, because the
transport does not guarantee ordering.*

### Usage and cost as JSONB maps

`{"input": 1200, "output": 340, "cache_read": 800, "reasoning": 512}` rather
than `input_tokens` / `output_tokens` columns. Providers add token categories
faster than you can migrate; prompt caching and reasoning tokens both appeared
after most schemas were written. Same argument for `cost_details`.

### `provided_*` vs resolved

The SDK may report a cost; the server may also compute one from the price
table. Both are stored. If your price table is wrong, you re-run the cost
resolver over `provided_usage_details` and fix history — without having
destroyed what the client actually sent.

### Trace totals computed on read

`traces` has no `total_cost` column. Aggregate from observations:

```sql
SELECT t.id, SUM(o.total_cost) AS cost, SUM((o.usage_details->>'input')::bigint) AS in_tok
  FROM traces t JOIN observations o
    ON o.project_id = t.project_id AND o.trace_id = t.id
 WHERE t.project_id = :pid
 GROUP BY t.id;
```

**When to change this:** when the trace-list endpoint gets slow — likely
somewhere past a few million observations. The fix is a denormalised
`total_cost` on `traces`, updated by the worker after each observation write,
which costs you a write amplification and a race to reason about. Do not do it
pre-emptively; *do* write the paragraph explaining the trade in your README,
because knowing when not to denormalise is the more valuable signal.

### The score model

Four fields carry the weight:

- **`name`** — many scores per trace, not one. "relevance", "toxicity", "cost_ok".
- **`data_type` + `value` / `string_value`** — numeric, categorical and boolean
  scores share one table.
- **`source`** — `API` | `EVAL` | `ANNOTATION`. Judge output and human labels
  in the same table with the same shape is what lets you measure judge-human
  agreement, which is how you show your judge is trustworthy. That measurement
  is a genuinely strong thing to put in a README.
- **`evaluator_version_id`** — which judge, which version. Without it, a score
  drop is ambiguous: did the agent get worse, or did the judge get stricter?

Note `source` is not on the wire protocol. It is derived server-side from the
credential, so an SDK cannot claim its score came from a human.

### The `project` relationship (and why the ORM is not for the worker)

Every tenant table declares `project: Mapped["Project"] = relationship()`. That
line exists for **insert ordering**, not navigation.

I hit this while testing the scaffold: with foreign keys but no `relationship()`
declarations, SQLAlchemy's unit of work has no mapper-level dependency graph, so
`session.add_all([project, trace, observation])` tries to insert the observation
first and dies on `observations_project_id_fkey`. Foreign keys tell *Postgres*
about the dependency; only relationships tell the *ORM*.

Consequence worth internalising: **the ingestion worker should not use the ORM.**
Use Core bulk upserts —

```python
from sqlalchemy.dialects.postgresql import insert
stmt = insert(Observation).values(rows)
stmt = stmt.on_conflict_do_update(index_elements=["project_id", "id"], set_={...})
```

— because you are writing hundreds of rows per batch and you need
`ON CONFLICT` semantics the ORM does not give you cleanly. Keep the ORM for the
read API and for tests, and never lazy-load through `.project` in a request path
(that is an N+1 waiting to happen).

---

## Verification already done

The scaffold was run against PostgreSQL 16 before you got it:

- 17 tables and 7 enum types create cleanly via `create_all`
- `latency_ms` and `time_to_first_token_ms` emit as
  `GENERATED ALWAYS AS (...) STORED` and compute correctly (3000.0 ms and a
  500.0 ms TTFT on the fixture trace)
- A three-level agent → {retriever, generation} tree round-trips through the
  recursive CTE with correct depth ordering
- Cost rolls up from `observations.total_cost`; input tokens roll up out of the
  `usage_details` JSONB map
- EVAL and ANNOTATION scores for the same metric aggregate side by side
- An observation for a trace that **does not exist yet** inserts successfully,
  confirming the out-of-order ingestion path works
- Every constraint rejects its case: score with no anchor, score with two
  anchors, categorical score with no `string_value`, observation ending before
  it starts, `sampling_rate` of 1.5, duplicate `(run, item)` pair, and a trace
  in a nonexistent project

What is **not** verified and is your job on Day 2: that Alembic autogenerate
produces this schema faithfully. It is unreliable with `Computed` columns and
check constraints — diff its output against `create_all` and reconcile.

---

## What you implement next

The schema is complete — schemas cannot be meaningfully stubbed. These are the
pieces to build on it, in order.

### 1. Alembic migration (Day 2)

```bash
alembic revision --autogenerate -m "v2: observation tree, tenancy, scores"
```

Then read every line of the generated file. Autogenerate does not handle
`Computed` columns or `CheckConstraint`s reliably — verify both. Check that
`latency_ms` comes out as `GENERATED ALWAYS AS (...) STORED`.

**Delete `init_db()` and the `create_all()` call in `worker.py`.** Two sources
of schema truth is the kind of thing that looks careless in a code review.

There is no v1→v2 data migration and you do not need one. Drop the old tables;
the data was synthetic benchmark traffic.

### 2. API key auth (Day 3)

```
Authorization: Basic base64(public_key ":" secret_key)
```

- Generate: `pk-ow-<22 chars>` / `sk-ow-<40 chars>` from `secrets.token_urlsafe`.
- Store `sha256(secret + per_key_salt)`. **Not bcrypt** — see the `ApiKey`
  docstring for why, and be ready to explain it; interviewers probe this.
- Cache `hash → {project_id, key_id}` in Redis with a TTL. Emit
  `cache_hit` / `cache_miss` counters — you will want them for the README.
- gRPC: a `grpc.aio.ServerInterceptor` that resolves `project_id` and puts it
  in a contextvar. FastAPI: a `Depends()` that returns an `AuthContext`.
- Update `last_used_at` asynchronously, not in the request path.

Write the negative tests first: no header, malformed base64, unknown public
key, wrong secret, expired key, and — the important one — **a valid key for
project A requesting a trace in project B returns 404, not 403.** Leaking
existence is an information disclosure bug.

### 3. Worker rewrite (Day 4)

Per event type:

- `trace_create` → `INSERT ... ON CONFLICT (project_id, id) DO UPDATE`
- `observation_create` → insert; if a stub already exists from an out-of-order
  child, merge rather than overwrite
- `observation_update` → **sparse patch**. Only overwrite columns present in
  the message. This is the subtle one: a naive `UPDATE SET everything` wipes
  fields the update event did not carry.
- `score_create` → insert, `source` from the credential

Idempotency: unique index on `event_id` in a small `processed_events` table, or
`ON CONFLICT DO NOTHING` keyed on it. At-least-once delivery means you *will*
see duplicates.

### 4. SDK v2 (Days 6–7)

The four v1 problems, and the fix for each:

| v1 problem | Fix |
|---|---|
| Fresh `trace_id` per call → nesting impossible | `contextvars.ContextVar` holding the current span stack |
| Blocking gRPC in the caller's hot path | `queue.Queue` + background flush thread, batch on size or interval |
| `model="unknown"`, `token_count=0` hardcoded | Read from the provider response; accept explicit kwargs |
| No try/except around the wrapped call → **errors invisible** | `except: set level=ERROR, status_message=repr(e); raise` |

Target API:

```python
ow = OpenWeave(public_key=..., secret_key=..., host=...)

@ow.observe(type="AGENT")
def handle(q):
    docs = retrieve(q)          # nested automatically via contextvar
    return answer(q, docs)

@ow.observe(type="GENERATION")
def answer(q, docs):
    r = client.chat.completions.create(...)
    ow.current().update(
        model=r.model,
        usage_details={"input": r.usage.prompt_tokens,
                       "output": r.usage.completion_tokens},
    )
    return r.choices[0].message.content
```

Sync and async both — most Python LLM code is sync, and v1 was async-only.
Ship an `atexit` flush, an explicit `ow.flush()`, and make the decorator a
no-op-with-a-warning if the transport is down. **An observability SDK must
never take down the app it observes.** Test that: point it at a dead port and
assert the decorated function still returns.

### 5. Eval loop (Day 10)

```
worker persists trace
      → XADD evals {trace_id, project_id}
      → eval consumer
      → load active EvaluationRule (filter + sampling_rate)
      → load EvaluatorVersion (prompt, model, variable_mapping, output_schema)
      → render prompt, call judge, validate against output_schema
      → INSERT scores (source=EVAL, evaluator_version_id=...)
      → UPDATE job_executions (status, judge_latency_ms, judge_cost)
```

Three things v1 did not do: it never triggered automatically at all, it ran the
judge synchronously inside the API request, and it recorded nothing when the
judge failed. `job_executions` is the table that fixes the third.

Sample production traffic (5–10%), evaluate dataset runs at 100%.

### 6. Experiments (Days 11–13)

```python
run = ow.create_run(dataset="support-qa", name="prompt-v7",
                    metadata={"prompt_version": 7})
for item in ow.get_dataset("support-qa").items:
    with run.item(item) as span:     # creates trace + dataset_run_item
        span.output = my_agent(item.input)
# scores land via the eval queue against dataset_run_item_id
```

Then the comparison query in the `DatasetRunItem` docstring, exposed as
`GET /datasets/{name}/runs/compare?runs=prompt-v6,prompt-v7`.

**This is your headline demo.** A table showing relevance up 0.08 and cost down
12% between two prompt versions is the single most compelling artifact this
project can produce for an AI infra role. Put its output in the README.

---

## Deliberately left out

| Left out | Why | Cost to add later |
|---|---|---|
| Organizations above projects | One tenancy level proves the pattern; an org layer buys an RBAC matrix you do not need | Table + nullable FK, ~1 day |
| Dataset item temporal versioning (`valid_from`/`valid_to`) | Correct, but adds a dimension to every dataset query | Add columns + widen PK; do it before you have real users, not after |
| `prompts` table | The `prompt_name`/`prompt_version` columns already exist on observations, so grouping runs by prompt version works without it | Table + link, ~1 day, no backfill needed |
| ClickHouse | A scale answer, and scale is explicitly not your gap. The schema *shape* is what matters and Postgres holds it fine at portfolio size | Significant — but say in your README that you know it is the next step and why |
| Annotation UI | The `ANNOTATION` score source and `score_configs` are already in the schema, so the data model supports it whenever you build the UI | UI only |
| SSO, billing, entitlements, feature flags | Business features, not engineering signal | n/a — do not |

Mention the left-out items in your README under "what I would build next, and
why I didn't". Knowing where to stop reads as judgment; building everything
badly reads as the opposite.

---

## Two things to fix in the repo this week, independent of the code

1. **The README claims automatic evaluation that does not exist.** v1 only
   evaluates via a manual `POST /traces/{id}/evaluate`. Fix the claim now — an
   interviewer who reads the code and catches the gap is a bad outcome, and it
   costs you more than the feature was worth.

2. **You built a benchmark suite and never published the numbers.** Almost no
   portfolio project measures anything. Run `benchmark/load_test.py`, put p50 /
   p95 / p99 and throughput in the README with the hardware and config, then
   re-run it after M1 and show both. "I measured it, changed the design, and
   measured again" is a stronger story than any single number.

Also: commit incrementally from here. The repo currently has one commit, which
tells a reader nothing about how you work.
