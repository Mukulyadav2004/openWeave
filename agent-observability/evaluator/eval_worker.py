"""Evaluation worker — consumes the `evals` stream and scores traces.

This is the piece v1 claimed to have and did not. In v1 the judge only ever ran
when a human called POST /traces/{id}/evaluate, it ran synchronously inside the
API request, its prompt was a constant in the source, and when it failed
nothing was recorded at all.

What runs here instead:

    worker commits a batch
      -> announces (project_id, trace_id) on the `evals` stream
      -> this worker waits until the trace has SETTLED
      -> loads the project's active EvaluationRules
      -> deterministic sampling + filter
      -> loads the pinned EvaluatorVersion (prompt, model, mapping, schema)
      -> renders, calls the judge, validates the verdict against the schema
      -> writes a Score (source=EVAL, evaluator_version_id=...)
      -> writes a JobExecution either way, with latency and error text

Three things are worth being able to explain:

SETTLING. A trace is announced every time any of its spans lands, so it is
announced many times and usually while still being written. Evaluating on first
sight would judge a half-finished trace. Each announcement therefore schedules a
check SETTLE_SECONDS out in a Redis sorted set, and every later announcement
pushes it back — a debounce. When a check comes due, the trace is judged only if
its most recent write is at least SETTLE_SECONDS old; otherwise it is checked
again later, up to a cap.

DETERMINISTIC SAMPLING. The sample decision is a hash of the trace id, not
random(). Because a trace is announced repeatedly, random() would give it a new
roll every time and a 5% rule would converge on evaluating everything.

VERSION PINNING. Every Score records the exact EvaluatorVersion that produced
it. Without that, a score drop is ambiguous: did the agent get worse, or did
someone edit the judge prompt?
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import time
from datetime import datetime, timezone

import redis.asyncio as redis
from redis.exceptions import ResponseError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import JudgeError, call_judge, render_prompt, validate_output  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("eval-worker")

# REDIS_URL carries the password when the host requires one (Railway's does).
REDIS_URL = os.getenv("REDIS_URL") or "redis://{}:{}".format(
    os.getenv("REDIS_HOST", "localhost"), os.getenv("REDIS_PORT", "6379"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://openweave:openweave@localhost:5432/openweave",
)
# Hosted Postgres hands out postgres:// or postgresql:// URLs; SQLAlchemy's
# asyncio engine needs the asyncpg driver named explicitly.
if DATABASE_URL.startswith(("postgres://", "postgresql://")):
    DATABASE_URL = "postgresql+asyncpg://" + DATABASE_URL.split("://", 1)[1]

EVAL_STREAM = os.getenv("EVAL_STREAM", "evals")
EVAL_GROUP = os.getenv("EVAL_GROUP", "evaluators")
CONSUMER_NAME = os.getenv("HOSTNAME") or socket.gethostname()

BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "50"))
# Also how often due settle checks are drained while the stream is idle.
BLOCK_MS = int(os.getenv("EVAL_BLOCK_MS", "1000"))
# How quiet a trace must be before we judge it.
SETTLE_SECONDS = float(os.getenv("EVAL_SETTLE_SECONDS", "5"))
# Cap on settle checks (each SETTLE_SECONDS apart), so a trace that never goes
# quiet is still evaluated eventually.
MAX_BOUNCES = int(os.getenv("EVAL_MAX_BOUNCES", "20"))
BOUNCE_KEY = "eval:bounces"
# Pending settle checks: member = [project_id, trace_id], score = due time.
DUE_KEY = "eval:due"
CLAIM_LEASE_SECONDS = float(os.getenv("EVAL_CLAIM_LEASE_SECONDS", "120"))
DONE_TTL = int(os.getenv("EVAL_DONE_TTL", "86400"))


# --------------------------------------------------------------------------- #
# Sampling and filtering
# --------------------------------------------------------------------------- #
def sample_hash(trace_id: str) -> float:
    """Stable value in [0,1) derived from the trace id.

    Deterministic on purpose — see the module docstring. The same trace always
    gets the same roll, so repeated announcements cannot smuggle it past a
    sampling rate.
    """
    digest = hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / 0x1_0000_0000


def matches_filter(trace: dict, clauses: list | None) -> bool:
    """Apply a rule's filter to a trace row.

    Evaluated in Python rather than compiled into SQL: rules per project are
    counted in tens, the trace row is already loaded, and generating SQL from
    user-supplied JSON is exactly where injection bugs live.
    """
    if not clauses:
        return True
    for clause in clauses:
        column = clause.get("column")
        op = clause.get("op", "=")
        want = clause.get("value")
        have = trace.get(column)

        if op == "=" and have != want:
            return False
        if op == "!=" and have == want:
            return False
        if op == "contains" and (have is None or str(want) not in str(have)):
            return False
        if op == "in" and have not in (want or []):
            return False
        if op == "exists" and (have is None) == bool(want):
            return False
    return True


# --------------------------------------------------------------------------- #
# Variable resolution
# --------------------------------------------------------------------------- #
async def build_variables(conn, trace: dict, mapping: dict | None) -> dict:
    """Resolve an evaluator's variable_mapping against this trace.

    Supported sources:
        trace.<column>            input, output, name, user_id, metadata, ...
        root.<column>             the root observation's column
        dataset_item.<column>     the expected output, when the trace came from
                                  a dataset run — this is what lets ONE judge
                                  template serve both live traffic and offline
                                  experiments
        literal:<text>            a constant
    """
    mapping = mapping or {"input": "trace.input", "output": "trace.output"}
    variables: dict = {}
    root = dataset_item = None

    for name, source in mapping.items():
        if not isinstance(source, str):
            variables[name] = source
            continue

        if source.startswith("literal:"):
            variables[name] = source[len("literal:"):]

        elif source.startswith("trace."):
            variables[name] = trace.get(source[len("trace."):])

        elif source.startswith("root."):
            if root is None:
                row = (await conn.execute(text("""
                    SELECT * FROM observations
                     WHERE project_id = :p AND trace_id = :t
                       AND parent_observation_id IS NULL
                     ORDER BY start_time LIMIT 1
                """), {"p": trace["project_id"], "t": trace["id"]})).mappings().first()
                root = dict(row) if row else {}
            variables[name] = root.get(source[len("root."):])

        elif source.startswith("dataset_item."):
            if dataset_item is None:
                row = (await conn.execute(text("""
                    SELECT di.* FROM dataset_run_items ri
                      JOIN dataset_items di
                        ON di.id = ri.dataset_item_id AND di.project_id = ri.project_id
                     WHERE ri.project_id = :p AND ri.trace_id = :t
                     LIMIT 1
                """), {"p": trace["project_id"], "t": trace["id"]})).mappings().first()
                dataset_item = dict(row) if row else {}
            variables[name] = dataset_item.get(source[len("dataset_item."):])

        else:
            variables[name] = source
    return variables


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
async def load_rules(conn, project_id: str) -> list[dict]:
    rows = (await conn.execute(text("""
        SELECT r.id AS rule_id, r.evaluator_id, r.filter, r.sampling_rate,
               v.id AS version_id, v.version, v.prompt, v.model, v.provider,
               v.model_params, v.variable_mapping, v.output_schema AS output_definition,
               v.score_name, v.score_data_type
          FROM evaluation_rules r
          JOIN LATERAL (
                SELECT * FROM evaluator_versions ev
                 WHERE ev.evaluator_id = r.evaluator_id
                 ORDER BY ev.version DESC LIMIT 1
               ) v ON TRUE
         WHERE r.project_id = :p AND r.is_active AND r.target = 'TRACE'
    """), {"p": project_id})).mappings().all()
    return [dict(r) for r in rows]


async def already_evaluated(redis_client, project_id, trace_id, rule_id) -> bool:
    """SETNX guard, keyed per rule so adding a rule re-evaluates old traces."""
    key = f"eval:done:{project_id}:{trace_id}:{rule_id}"
    claimed = await redis_client.set(key, "1", nx=True, ex=DONE_TTL)
    return not claimed


async def run_rule(conn, trace: dict, rule: dict, judge_fn) -> str:
    """Evaluate one trace under one rule. Returns a short outcome label.

    `judge_fn` is injected so tests can run the whole path — mapping, rendering,
    validation, score and job rows — without an LLM.
    """
    now = datetime.now(timezone.utc)
    job_id = f"{trace['id']}:{rule['rule_id']}:{int(now.timestamp() * 1000)}"

    await conn.execute(text("""
        INSERT INTO job_executions
            (project_id, id, evaluator_version_id, rule_id, trace_id,
             status, retry_count, created_at, started_at)
        VALUES (:p, :id, :v, :r, :t, 'RUNNING', 0, :now, :now)
    """), {"p": trace["project_id"], "id": job_id, "v": rule["version_id"],
           "r": rule["rule_id"], "t": trace["id"], "now": now})

    started = time.perf_counter()
    try:
        variables = await build_variables(conn, trace, rule["variable_mapping"])
        prompt = render_prompt(rule["prompt"], variables)
        verdict = await judge_fn(
            rule["provider"] or "ollama", rule["model"], prompt,
            rule["model_params"] or {},
        )
        verdict = validate_output(verdict, rule["output_definition"])

        data_type = rule["score_data_type"] or "NUMERIC"
        raw = verdict.get("score")
        value = string_value = None
        if data_type == "CATEGORICAL":
            string_value = str(raw)
        elif data_type == "BOOLEAN":
            value = 1.0 if raw in (True, "true", 1, "1") else 0.0
        else:
            value = float(raw)

        score_id = f"{job_id}:score"
        await conn.execute(text("""
            INSERT INTO scores
                (project_id, id, trace_id, name, data_type, value, string_value,
                 source, comment, evaluator_version_id, timestamp, created_at)
            VALUES (:p, :id, :t, :name, CAST(:dt AS score_data_type), :val, :sval,
                    'EVAL', :comment, :v, :now, :now)
            ON CONFLICT (project_id, id) DO NOTHING
        """), {"p": trace["project_id"], "id": score_id, "t": trace["id"],
               "name": rule["score_name"], "dt": data_type, "val": value,
               "sval": string_value, "comment": verdict.get("reason"),
               "v": rule["version_id"], "now": datetime.now(timezone.utc)})

        await conn.execute(text("""
            UPDATE job_executions
               SET status='COMPLETED', score_id=:s, judge_latency_ms=:ms, ended_at=:now
             WHERE project_id=:p AND id=:id
        """), {"s": score_id, "ms": (time.perf_counter() - started) * 1000,
               "now": datetime.now(timezone.utc), "p": trace["project_id"],
               "id": job_id})
        return f"scored {rule['score_name']}={value if value is not None else string_value}"

    except (JudgeError, ValueError, TypeError, KeyError) as exc:
        # The failure is RECORDED rather than swallowed. "We have no score and
        # here is exactly why" is the difference between a platform and a script.
        await conn.execute(text("""
            UPDATE job_executions
               SET status='ERROR', error=:e, judge_latency_ms=:ms, ended_at=:now
             WHERE project_id=:p AND id=:id
        """), {"e": str(exc)[:4000], "ms": (time.perf_counter() - started) * 1000,
               "now": datetime.now(timezone.utc), "p": trace["project_id"],
               "id": job_id})
        logger.warning("eval failed trace=%s rule=%s: %s",
                       trace["id"], rule["rule_id"], exc)
        return f"error: {exc}"


async def evaluate_trace(conn, redis_client, project_id: str, trace_id: str,
                         judge_fn=call_judge) -> list[str]:
    """Run every applicable rule against one trace. Returns outcome labels."""
    trace_row = (await conn.execute(text(
        "SELECT * FROM traces WHERE project_id = :p AND id = :t"
    ), {"p": project_id, "t": trace_id})).mappings().first()
    if trace_row is None:
        return ["trace not found"]
    trace = dict(trace_row)

    outcomes = []
    for rule in await load_rules(conn, project_id):
        if sample_hash(trace_id) >= float(rule["sampling_rate"]):
            continue
        if not matches_filter(trace, rule["filter"]):
            continue
        if await already_evaluated(redis_client, project_id, trace_id, rule["rule_id"]):
            continue
        outcomes.append(await run_rule(conn, trace, rule, judge_fn))
    return outcomes


# --------------------------------------------------------------------------- #
# Settling
# --------------------------------------------------------------------------- #
async def is_settled(conn, project_id: str, trace_id: str) -> bool:
    last = (await conn.execute(text("""
        SELECT GREATEST(
                 COALESCE(MAX(o.updated_at), to_timestamp(0)),
                 COALESCE(MAX(t.updated_at), to_timestamp(0))
               )
          FROM traces t
          LEFT JOIN observations o
            ON o.project_id = t.project_id AND o.trace_id = t.id
         WHERE t.project_id = :p AND t.id = :t
    """), {"p": project_id, "t": trace_id})).scalar()
    if last is None:
        return False
    age = (datetime.now(timezone.utc) - last).total_seconds()
    return age >= SETTLE_SECONDS


# --------------------------------------------------------------------------- #
# Stream loop
# --------------------------------------------------------------------------- #
async def ensure_group(redis_client) -> None:
    try:
        await redis_client.xgroup_create(
            name=EVAL_STREAM, groupname=EVAL_GROUP, id="0", mkstream=True
        )
        logger.info("created consumer group %s", EVAL_GROUP)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def _due_member(project_id: str, trace_id: str) -> str:
    return json.dumps([project_id, trace_id])


async def handle_message(redis_client, message_id, fields) -> None:
    """Schedule a settle check for an announced trace. Never evaluates.

    Each announcement moves the check to SETTLE_SECONDS from now (GT: never
    earlier), so a trace still receiving spans keeps being deferred. Re-XADDing
    unsettled traces onto the stream instead does not work: this loop re-reads
    them within milliseconds, the bounce cap runs out in ~100ms, and the
    half-written trace is judged anyway.
    """
    project_id, trace_id = fields.get("project_id"), fields.get("trace_id")
    if project_id and trace_id:
        await redis_client.zadd(
            DUE_KEY, {_due_member(project_id, trace_id): time.time() + SETTLE_SECONDS},
            gt=True,
        )
    await redis_client.xack(EVAL_STREAM, EVAL_GROUP, message_id)


async def drain_due(engine, redis_client, judge_fn=call_judge) -> int:
    """Evaluate traces whose settle check is due. Returns how many were checked."""
    now = time.time()
    members = await redis_client.zrangebyscore(
        DUE_KEY, "-inf", now, start=0, num=BATCH_SIZE
    )
    for member in members:
        # Lease the entry before working on it, so a crash mid-evaluation is
        # retried once the lease expires. Two workers racing on the same entry
        # are still deduplicated by the per-rule SETNX in evaluate_trace.
        await redis_client.zadd(DUE_KEY, {member: now + CLAIM_LEASE_SECONDS}, xx=True)
        project_id, trace_id = json.loads(member)
        bounce_key = f"{project_id}:{trace_id}"

        async with engine.begin() as conn:
            settled = await is_settled(conn, project_id, trace_id)
        if not settled:
            bounces = await redis_client.hincrby(BOUNCE_KEY, bounce_key, 1)
            if bounces <= MAX_BOUNCES:
                await redis_client.zadd(DUE_KEY, {member: time.time() + SETTLE_SECONDS})
                continue
            logger.warning("trace %s never settled after %d checks; evaluating anyway",
                           trace_id, bounces)

        async with engine.begin() as conn:
            outcomes = await evaluate_trace(
                conn, redis_client, project_id, trace_id, judge_fn=judge_fn
            )
        await redis_client.zrem(DUE_KEY, member)
        await redis_client.hdel(BOUNCE_KEY, bounce_key)
        for outcome in outcomes:
            logger.info("trace=%s %s", trace_id, outcome)
    return len(members)


async def run() -> None:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    engine = create_async_engine(DATABASE_URL, pool_size=5, max_overflow=5)

    await ensure_group(redis_client)
    logger.info("eval worker %s consuming %s (settle=%.1fs)",
                CONSUMER_NAME, EVAL_STREAM, SETTLE_SECONDS)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    try:
        while not stop.is_set():
            response = await redis_client.xreadgroup(
                groupname=EVAL_GROUP, consumername=CONSUMER_NAME,
                streams={EVAL_STREAM: ">"}, count=BATCH_SIZE, block=BLOCK_MS,
            )
            for _stream, messages in response or []:
                for message_id, fields in messages:
                    try:
                        await handle_message(redis_client, message_id, fields)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("eval message %s failed: %s", message_id, exc)
                        await redis_client.xack(EVAL_STREAM, EVAL_GROUP, message_id)
            try:
                await drain_due(engine, redis_client)
            except Exception as exc:  # noqa: BLE001
                # Never let one bad trace kill the loop — that was one of v1's
                # failure modes and it is not worth repeating here. The entry
                # stays leased and is retried when the lease expires.
                logger.error("eval drain failed: %s", exc)
    finally:
        logger.info("shutting down")
        await redis_client.aclose()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())
