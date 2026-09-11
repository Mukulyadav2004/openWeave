"""Redis Streams worker — v2.

Consumes events from the `events` stream and writes them to Postgres.

Every v1 reliability bug is fixed here, and each fix is worth being able to
explain in an interview:

  1. CONSUMER NAME FROM HOSTNAME.  v1 hardcoded "worker-1", so two pods claimed
     the same consumer identity in the group and fought over one pending-entries
     list. (v1's k8s manifest said replicas: 1, which hid the bug.)

  2. XAUTOCLAIM RECOVERY.  v1 had none: if a worker died between XREADGROUP and
     XACK, that message sat in the PEL forever and nothing ever reclaimed it.
     Here, stranded messages idle longer than CLAIM_MIN_IDLE_MS get picked up by
     whichever worker notices first.

  3. DEAD LETTER QUEUE.  v1 acked malformed messages and logged them, i.e. threw
     the data away, and a non-IntegrityError DB failure propagated up and KILLED
     THE WORKER LOOP. Here a failing message is retried, then parked in
     ingestion_dead_letters and acked, and the loop survives.

  4. GRACEFUL SHUTDOWN.  v1 had no SIGTERM handler, so every rolling deploy
     dropped whatever was in flight.

  5. CORE BULK UPSERTS, NOT THE ORM.  Hundreds of rows per batch with ON
     CONFLICT semantics the ORM does not give you cleanly.

Out-of-order delivery is handled in two places, because over a queue it is
normal rather than exceptional:
  * an observation whose trace has not arrived yet -> a stub trace is upserted
  * an observation_update that arrives before its create -> the update inserts
    a partial row, which the later create fills in (COALESCE, never clobber)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import sys
from datetime import datetime, timezone

import redis.asyncio as redis
from redis.exceptions import ResponseError
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine

# shared/ holds code used by more than one service (auth, pricing). Adding it
# to sys.path here beats keeping duplicate copies per service in sync.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))

from models import IngestionDeadLetter, Observation, Score, Trace  # noqa: E402
from pricing import ModelPriceCache, resolve_costs  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("worker")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://openweave:openweave@localhost:5432/openweave",
)

STREAM_NAME = os.getenv("STREAM_NAME", "events")
GROUP_NAME = os.getenv("GROUP_NAME", "workers")
# Fix #1. In k8s, HOSTNAME is the pod name, which is unique per replica.
CONSUMER_NAME = os.getenv("HOSTNAME") or socket.gethostname()

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
BLOCK_MS = int(os.getenv("BLOCK_MS", "2000"))
# A message idle this long is assumed orphaned by a dead worker.
CLAIM_MIN_IDLE_MS = int(os.getenv("CLAIM_MIN_IDLE_MS", "60000"))
CLAIM_EVERY_N_LOOPS = int(os.getenv("CLAIM_EVERY_N_LOOPS", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_KEY = "worker:retries"

# Traces whose ingest finished get announced here; the eval worker consumes it.
EVAL_STREAM = os.getenv("EVAL_STREAM", "evals")
EVAL_STREAM_MAXLEN = int(os.getenv("EVAL_STREAM_MAXLEN", "200000"))
EVAL_ENQUEUE = os.getenv("EVAL_ENQUEUE", "1") == "1"

# Shared across batches: the price tables change roughly never, and a per-batch
# reload would put two queries in front of every write.
PRICE_CACHE = ModelPriceCache(ttl_seconds=int(os.getenv("PRICE_CACHE_TTL", "300")))


# --------------------------------------------------------------------------- #
# Conversion helpers
#
# MessageToDict gives us JSON types, so everything needs coercing back:
#   * timestamps are RFC3339 strings
#   * uint64 map values come back as STRINGS (protobuf's JSON mapping does this
#     deliberately, because JSON numbers cannot hold 64 bits safely)
#   * enums come back as their full proto names, e.g. OBSERVATION_TYPE_TOOL
# --------------------------------------------------------------------------- #
def _ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _enum(value, prefix, default=None):
    if not value:
        return default
    name = value[len(prefix):] if value.startswith(prefix) else value
    return default if name == "UNSPECIFIED" else name


def _usage(value):
    if not value:
        return None
    return {k: int(v) for k, v in value.items()}


def _cost(value):
    if not value:
        return None
    return {k: float(v) for k, v in value.items()}


def _present(body: dict, *keys) -> dict:
    """Only the keys actually sent — the basis of the sparse patch."""
    return {k: body[k] for k in keys if k in body}


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #
TRACE_COLUMNS = (
    "project_id", "id", "name", "user_id", "session_id", "timestamp",
    "input", "output", "metadata", "tags", "release", "version",
    "is_experiment", "created_at", "updated_at",
)


def build_trace_row(event: dict) -> dict:
    body, now = event["body"], datetime.now(timezone.utc)
    return {
        "project_id": event["project_id"],
        "id": body["id"],
        "name": body.get("name"),
        "user_id": body.get("user_id"),
        "session_id": body.get("session_id"),
        "timestamp": _ts(body.get("timestamp")) or _ts(event.get("ts")) or now,
        "input": body.get("input"),
        "output": body.get("output"),
        "metadata": body.get("metadata"),
        "tags": body.get("tags"),
        "release": body.get("release"),
        "version": body.get("version"),
        "is_experiment": bool(body.get("is_experiment", False)),
        "created_at": now,
        "updated_at": now,
    }


OBSERVATION_COLUMNS = (
    "project_id", "id", "trace_id", "parent_observation_id", "type", "name",
    "start_time", "end_time", "completion_start_time", "level",
    "status_message", "input", "output", "metadata", "provided_model_name",
    "model_parameters", "provided_usage_details", "usage_details",
    "provided_cost_details", "cost_details", "total_cost", "prompt_name",
    "prompt_version", "created_at", "updated_at",
)


def _observation_fields(event: dict) -> dict:
    """Fields the client actually sent, already coerced. Never includes keys
    that were absent from the message — that is what makes updates sparse."""
    body = event["body"]
    out: dict = {}

    for key in ("parent_observation_id", "name", "status_message", "input",
                "output", "metadata", "model_parameters", "prompt_name"):
        if key in body:
            out[key] = body[key]

    if "model" in body:
        out["provided_model_name"] = body["model"]
    if "prompt_version" in body:
        out["prompt_version"] = int(body["prompt_version"])
    if "type" in body:
        out["type"] = _enum(body["type"], "OBSERVATION_TYPE_")
    if "level" in body:
        out["level"] = _enum(body["level"], "OBSERVATION_LEVEL_", "DEFAULT")

    for key in ("start_time", "end_time", "completion_start_time"):
        if key in body:
            out[key] = _ts(body[key])

    if "usage_details" in body:
        usage = _usage(body["usage_details"])
        out["provided_usage_details"] = usage
        # Resolved usage == provided usage until the cost engine lands (Day 9).
        out["usage_details"] = usage
    if "cost_details" in body:
        cost = _cost(body["cost_details"])
        out["provided_cost_details"] = cost
        out["cost_details"] = cost
        out["total_cost"] = sum(cost.values()) if cost else None

    return out


def build_observation_row(event: dict) -> dict:
    body, now = event["body"], datetime.now(timezone.utc)
    row = {
        "project_id": event["project_id"],
        "id": body["id"],
        "trace_id": body["trace_id"],
        "created_at": now,
        "updated_at": now,
        **_observation_fields(event),
    }
    # NOT NULL columns need a value on the INSERT path even for an update that
    # arrives before its create.
    row.setdefault("type", "SPAN")
    row.setdefault("level", "DEFAULT")
    # Prefer end_time over the event timestamp: an update is stamped just AFTER
    # the span ended, and Postgres evaluates ck_obs_end_after_start against the
    # proposed INSERT row even when ON CONFLICT turns it into an UPDATE. Using
    # the event time rejected every update that carried end_time. A later
    # create overwrites this placeholder.
    row.setdefault("start_time", row.get("end_time") or _ts(event.get("ts")) or now)
    return row


def build_score_row(event: dict) -> dict:
    body, now = event["body"], datetime.now(timezone.utc)
    return {
        "project_id": event["project_id"],
        "id": body["id"],
        "trace_id": body.get("trace_id"),
        "observation_id": body.get("observation_id"),
        "dataset_run_item_id": body.get("dataset_run_item_id"),
        "name": body["name"],
        "data_type": _enum(body.get("data_type"), "SCORE_DATA_TYPE_", "NUMERIC"),
        "value": float(body["value"]) if body.get("value") is not None else None,
        "string_value": body.get("string_value"),
        # Derived from the credential, never from the client. An SDK key can
        # only ever produce API scores; EVAL comes from the eval worker and
        # ANNOTATION from an authenticated human session.
        "source": "API",
        "comment": body.get("comment"),
        "timestamp": _ts(event.get("ts")) or now,
        "created_at": now,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _merge_upsert(table, sent, key):
    """INSERT ... ON CONFLICT DO UPDATE that MERGES rather than clobbers.

    Only the columns the event actually sent are updated. Row builders fill NOT
    NULL columns (timestamp, is_experiment, level) with placeholders so the
    INSERT path works, and those placeholders must never reach the UPDATE path:
    a trace_create carrying only `output` would otherwise reset the trace's
    timestamp and is_experiment. COALESCE(excluded.col, table.col) additionally
    means a sent null cannot wipe what an earlier event established.
    """
    stmt = insert(table)
    updatable = (set(sent) | {"updated_at"}) - set(key) - {"created_at"}
    return stmt.on_conflict_do_update(
        index_elements=list(key),
        set_={
            c: func.coalesce(stmt.excluded[c], table.__table__.c[c])
            for c in sorted(updatable)
        },
    )


def _group_by_sent(pairs):
    """executemany needs one statement per SET clause, so group rows by the set
    of columns their event carried. SDKs emit only a handful of shapes."""
    groups: dict[frozenset, list[dict]] = {}
    for row, sent in pairs:
        groups.setdefault(frozenset(sent), []).append(row)
    return groups


async def apply_events(conn, events: list[dict], price_cache=None) -> dict:
    """Write one batch. Pure DB logic, no Redis — which makes it unit testable.

    Ordering within the batch matters: traces (real and stub) go first so the
    trace list never shows an observation whose parent row does not exist yet.

    Returns the keys it touched, so the caller can price the generations and
    announce the traces for evaluation without re-deriving them.
    """
    key = ("project_id", "id")
    traces, obs_create, obs_update, scores = [], [], [], []
    for event in events:
        kind = event.get("type")
        if kind == "trace_create":
            sent = {c for c in TRACE_COLUMNS if c in event["body"]}
            traces.append((build_trace_row(event), sent))
        elif kind == "observation_create":
            # A create is authoritative for type and start_time; every other
            # column is overwritten only if this event carried it.
            sent = set(_observation_fields(event)) | {"type", "start_time"}
            obs_create.append((build_observation_row(event), sent))
        elif kind == "observation_update":
            obs_update.append(event)
        elif kind == "score_create":
            scores.append(build_score_row(event))

    # --- stub traces for observations whose trace has not arrived ---------- #
    referenced = {
        (e["project_id"], e["body"]["trace_id"])
        for e in events
        if e.get("type", "").startswith("observation_")
    }
    known = {(r["project_id"], r["id"]) for r, _ in traces}
    now = datetime.now(timezone.utc)
    stubs = [
        {
            "project_id": pid, "id": tid, "name": None, "user_id": None,
            "session_id": None, "timestamp": now, "input": None, "output": None,
            "metadata": None, "tags": None, "release": None, "version": None,
            "is_experiment": False, "created_at": now, "updated_at": now,
        }
        for pid, tid in referenced - known
    ]

    for sent, rows in _group_by_sent(traces).items():
        await conn.execute(_merge_upsert(Trace, sent, key), rows)
    if stubs:
        # A stub only fills a gap. DO NOTHING rather than merge: its placeholder
        # timestamp and is_experiment=False would otherwise overwrite a real
        # trace that arrived in an earlier batch.
        await conn.execute(
            insert(Trace).on_conflict_do_nothing(index_elements=list(key)), stubs
        )

    for sent, rows in _group_by_sent(obs_create).items():
        normalised = [{c: r.get(c) for c in OBSERVATION_COLUMNS} for r in rows]
        await conn.execute(_merge_upsert(Observation, sent, key), normalised)

    # Updates are applied one at a time ON PURPOSE: each carries a different
    # set of columns, and executemany requires uniform parameter keys. Updates
    # are a fraction of ingest volume, so this is not the hot path.
    for event in obs_update:
        row = build_observation_row(event)
        touched = set(_observation_fields(event)) | {"updated_at"}
        stmt = insert(Observation).values(**row)
        await conn.execute(
            stmt.on_conflict_do_update(
                index_elements=["project_id", "id"],
                set_={c: stmt.excluded[c] for c in touched},
            )
        )

    if scores:
        stmt = insert(Score)
        await conn.execute(
            stmt.on_conflict_do_nothing(index_elements=["project_id", "id"]), scores
        )

    touched_observations = {
        (r["project_id"], r["id"]) for r, _ in obs_create
    } | {
        (e["project_id"], e["body"]["id"]) for e in obs_update
    }

    # Cost is resolved AFTER the write, in the same transaction, because a
    # generation's model name and its token usage usually arrive in different
    # events (model on the create, usage on the update). Pricing during row
    # building would score half of them at zero.
    if price_cache is not None and touched_observations:
        await resolve_costs(conn, price_cache, touched_observations)

    return {
        "observations": touched_observations,
        "traces": {(r["project_id"], r["id"]) for r, _ in traces} | referenced,
    }


# --------------------------------------------------------------------------- #
# Stream plumbing
# --------------------------------------------------------------------------- #
async def ensure_group(redis_client) -> None:
    try:
        await redis_client.xgroup_create(
            name=STREAM_NAME, groupname=GROUP_NAME, id="0", mkstream=True
        )
        logger.info("created consumer group %s", GROUP_NAME)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def dead_letter(engine, redis_client, message_id, raw, error) -> None:
    """Park a message we cannot process, then ack it so the PEL stays clean."""
    try:
        payload = json.loads(raw.get("data", "{}"))
    except json.JSONDecodeError:
        payload = {"unparseable": raw.get("data", "")[:4000]}
    try:
        async with engine.begin() as conn:
            await conn.execute(
                insert(IngestionDeadLetter).values(
                    project_id=payload.get("project_id"),
                    event_id=payload.get("event_id"),
                    stream_message_id=message_id,
                    payload=payload,
                    error=str(error)[:4000],
                    retry_count=MAX_RETRIES,
                    created_at=datetime.now(timezone.utc),
                )
            )
    except Exception as exc:  # noqa: BLE001
        # If even the DLQ write fails, log loudly and ack anyway — a poison
        # message must never be able to wedge the consumer permanently.
        logger.error("DLQ write failed for %s: %s", message_id, exc)
    await redis_client.xack(STREAM_NAME, GROUP_NAME, message_id)
    await redis_client.hdel(RETRY_KEY, message_id)
    logger.warning("dead-lettered %s: %s", message_id, error)


async def handle_batch(engine, redis_client, messages) -> None:
    """Try the batch as one transaction; on failure, isolate the bad message."""
    parsed, unparseable = [], []
    for message_id, fields in messages:
        try:
            parsed.append((message_id, json.loads(fields["data"])))
        except (KeyError, json.JSONDecodeError) as exc:
            unparseable.append((message_id, fields, exc))

    for message_id, fields, exc in unparseable:
        await dead_letter(engine, redis_client, message_id, fields, exc)

    if not parsed:
        return

    try:
        async with engine.begin() as conn:
            touched = await apply_events(
                conn, [event for _, event in parsed], PRICE_CACHE
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("batch of %d failed (%s); retrying individually",
                       len(parsed), exc)
        # One bad event must not cost the whole batch, so fall back to
        # per-message writes to find it.
        for message_id, event in parsed:
            try:
                async with engine.begin() as conn:
                    one = await apply_events(conn, [event], PRICE_CACHE)
            except Exception as inner:  # noqa: BLE001
                count = await redis_client.hincrby(RETRY_KEY, message_id, 1)
                if count >= MAX_RETRIES:
                    await dead_letter(
                        engine, redis_client, message_id, {"data": json.dumps(event)},
                        inner,
                    )
                else:
                    logger.warning("message %s failed (attempt %d/%d): %s",
                                   message_id, count, MAX_RETRIES, inner)
                continue
            await redis_client.xack(STREAM_NAME, GROUP_NAME, message_id)
            await redis_client.hdel(RETRY_KEY, message_id)
            await announce_for_eval(redis_client, one["traces"])
        return

    ids = [message_id for message_id, _ in parsed]
    await redis_client.xack(STREAM_NAME, GROUP_NAME, *ids)
    if ids:
        await redis_client.hdel(RETRY_KEY, *ids)
    await announce_for_eval(redis_client, touched["traces"])
    logger.info("committed %d event(s)", len(ids))


async def announce_for_eval(redis_client, traces: set[tuple[str, str]]) -> None:
    """Tell the eval worker these traces received data.

    Announced AFTER the commit, never before: a trace the eval worker cannot
    read yet would just bounce around its settle loop. Duplicates are expected
    and cheap — one message per batch per trace — and the eval worker dedupes
    with a SETNX, because a trace's spans arrive across many batches.
    """
    if not (EVAL_ENQUEUE and traces):
        return
    try:
        async with redis_client.pipeline(transaction=False) as pipe:
            for project_id, trace_id in traces:
                pipe.xadd(
                    EVAL_STREAM,
                    {"project_id": project_id, "trace_id": trace_id},
                    maxlen=EVAL_STREAM_MAXLEN,
                    approximate=True,
                )
            await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        # Evaluation is best-effort relative to ingestion. Losing an eval
        # announcement must never cost us the trace itself.
        logger.warning("failed to announce %d trace(s) for eval: %s",
                       len(traces), exc)


async def reclaim_stranded(engine, redis_client) -> int:
    """Fix #2: adopt messages a dead worker left in the pending-entries list."""
    try:
        _cursor, messages, _ = await redis_client.xautoclaim(
            name=STREAM_NAME,
            groupname=GROUP_NAME,
            consumername=CONSUMER_NAME,
            min_idle_time=CLAIM_MIN_IDLE_MS,
            count=BATCH_SIZE,
        )
    except ResponseError as exc:
        logger.debug("xautoclaim unavailable: %s", exc)
        return 0
    if messages:
        logger.info("reclaimed %d stranded message(s)", len(messages))
        await handle_batch(engine, redis_client, messages)
    return len(messages)


async def run() -> None:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    engine = create_async_engine(DATABASE_URL, pool_size=5, max_overflow=5)

    await ensure_group(redis_client)
    logger.info(
        "worker %s consuming stream=%s group=%s batch=%d",
        CONSUMER_NAME, STREAM_NAME, GROUP_NAME, BATCH_SIZE,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    loops = 0
    try:
        while not stop.is_set():
            loops += 1
            if loops % CLAIM_EVERY_N_LOOPS == 0:
                await reclaim_stranded(engine, redis_client)

            response = await redis_client.xreadgroup(
                groupname=GROUP_NAME,
                consumername=CONSUMER_NAME,
                streams={STREAM_NAME: ">"},
                count=BATCH_SIZE,
                block=BLOCK_MS,
            )
            for _stream, messages in response or []:
                await handle_batch(engine, redis_client, messages)
    finally:
        # Fix #4: finish the batch in hand, then release cleanly. Anything still
        # unacked stays in the PEL and another worker reclaims it.
        logger.info("shutting down")
        await redis_client.aclose()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())
