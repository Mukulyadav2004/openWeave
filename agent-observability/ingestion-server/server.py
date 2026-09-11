"""gRPC ingestion server — v2.

Accepts authenticated BATCHES of events and pushes them onto a Redis Stream.
Does no database writes of its own: the whole job is authenticate, validate,
enqueue, acknowledge. Keeping it that thin is what lets it stay fast and what
lets the worker be restarted without dropping traffic.

Changes from v1 that are worth knowing:

  * Batched.  v1 was one RPC per trace with the caller blocked on it.
  * Authenticated.  v1 called add_insecure_port() with no interceptor at all —
    anyone who could reach :50051 could write traces.
  * project_id is injected server-side from the credential and is NOT
    accepted from the client, so a caller cannot write into another project.
  * Partial failure.  One malformed event returns an error for that event;
    the other 199 in the batch still land.
  * Bounded stream.  XADD MAXLEN caps Redis memory. v1 grew without limit.
  * Real health RPC for k8s probes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys

import asyncpg
import grpc
import redis.asyncio as redis
from google.protobuf.json_format import MessageToDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from auth import STATS, ApiKeyVerifier, AuthError  # noqa: E402

import openweave_pb2  # noqa: E402
import openweave_pb2_grpc  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ingestion-server")

VERSION = "2.0.0"
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
# 0.0.0.0 is reachable from every container runtime; a host that routes to
# containers over IPv6 (Railway's private network) needs GRPC_BIND=:: instead.
GRPC_BIND = os.getenv("GRPC_BIND", "0.0.0.0")
# REDIS_URL carries the password when the host requires one (Railway's does).
REDIS_URL = os.getenv("REDIS_URL") or "redis://{}:{}".format(
    os.getenv("REDIS_HOST", "localhost"), os.getenv("REDIS_PORT", "6379"))
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://openweave:openweave@localhost:5432/openweave"
).replace("+asyncpg", "").replace("+psycopg2", "")

STREAM_NAME = os.getenv("STREAM_NAME", "events")
# Cap the stream so a wedged worker cannot OOM Redis. ~ makes trimming cheap
# (Redis trims on node boundaries rather than exactly).
STREAM_MAXLEN = int(os.getenv("STREAM_MAXLEN", "1000000"))
MAX_BATCH_EVENTS = int(os.getenv("MAX_BATCH_EVENTS", "1000"))

# Maps the protobuf oneof field name to the string the worker dispatches on.
BODY_FIELDS = (
    "trace_create",
    "observation_create",
    "observation_update",
    "score_create",
)


def _to_dict(message) -> dict:
    """Protobuf -> dict, preserving field presence.

    including_default_value_fields is deliberately NOT set: unset `optional`
    fields must be ABSENT from the dict so the worker can apply
    observation_update as a sparse patch instead of overwriting with defaults.
    """
    return MessageToDict(message, preserving_proto_field_name=True)


def _validate(event, body_field: str, body: dict) -> str | None:
    """Return an error string, or None if the event is acceptable."""
    if not event.event_id:
        return "event_id is required"
    if body_field in ("trace_create",):
        if not body.get("id"):
            return "trace.id is required"
    elif body_field in ("observation_create", "observation_update"):
        if not body.get("id"):
            return "observation.id is required"
        if not body.get("trace_id"):
            return "observation.trace_id is required"
        if body_field == "observation_create" and not body.get("type"):
            return "observation.type is required on create"
    elif body_field == "score_create":
        if not body.get("id"):
            return "score.id is required"
        if not body.get("name"):
            return "score.name is required"
        anchors = sum(
            1 for k in ("trace_id", "dataset_run_item_id") if body.get(k)
        )
        if anchors != 1:
            return "score must anchor to exactly one of trace_id, dataset_run_item_id"
    return None


class IngestionService(openweave_pb2_grpc.IngestionServiceServicer):
    def __init__(self, redis_client, verifier: ApiKeyVerifier):
        self._redis = redis_client
        self._verifier = verifier

    async def _authenticate(self, context):
        header = None
        for key, value in context.invocation_metadata():
            if key.lower() == "authorization":
                header = value
                break
        try:
            return await self._verifier.verify(header)
        except AuthError as exc:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))

    async def IngestBatch(self, request, context):
        auth = await self._authenticate(context)

        if len(request.events) > MAX_BATCH_EVENTS:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"batch too large: {len(request.events)} > {MAX_BATCH_EVENTS}",
            )

        errors: list[openweave_pb2.EventError] = []
        payloads: list[str] = []

        for event in request.events:
            body_field = event.WhichOneof("body")
            if body_field not in BODY_FIELDS:
                errors.append(
                    openweave_pb2.EventError(
                        event_id=event.event_id, code=400, message="unknown event body"
                    )
                )
                continue

            body = _to_dict(getattr(event, body_field))
            problem = _validate(event, body_field, body)
            if problem:
                errors.append(
                    openweave_pb2.EventError(
                        event_id=event.event_id, code=400, message=problem
                    )
                )
                continue

            payloads.append(
                json.dumps(
                    {
                        "event_id": event.event_id,
                        # Injected from the credential, never from the client.
                        "project_id": auth.project_id,
                        "type": body_field,
                        "ts": event.timestamp.ToJsonString()
                        if event.HasField("timestamp")
                        else None,
                        "body": body,
                    }
                )
            )

        if payloads:
            try:
                # One round trip for the whole batch.
                async with self._redis.pipeline(transaction=False) as pipe:
                    for payload in payloads:
                        pipe.xadd(
                            STREAM_NAME,
                            {"data": payload},
                            maxlen=STREAM_MAXLEN,
                            approximate=True,
                        )
                    await pipe.execute()
            except Exception as exc:  # noqa: BLE001
                logger.error("failed to enqueue batch: %s", exc)
                await context.abort(
                    grpc.StatusCode.UNAVAILABLE, "ingestion queue unavailable"
                )

        # last_used_at is intentionally not awaited in the request path.
        asyncio.create_task(self._verifier.touch_last_used(auth.api_key_id))

        logger.info(
            "project=%s sdk=%s accepted=%d rejected=%d",
            auth.project_id,
            request.sdk or "unknown",
            len(payloads),
            len(errors),
        )
        return openweave_pb2.IngestBatchResponse(
            accepted=len(payloads), errors=errors
        )

    async def Health(self, request, context):
        redis_ok = True
        try:
            await self._redis.ping()
        except Exception:  # noqa: BLE001
            redis_ok = False
        return openweave_pb2.HealthResponse(
            ok=redis_ok, version=VERSION, redis_ok=redis_ok
        )


async def serve() -> None:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    pool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=2, max_size=10)
    verifier = ApiKeyVerifier(pool, redis_client)

    server = grpc.aio.server()
    openweave_pb2_grpc.add_IngestionServiceServicer_to_server(
        IngestionService(redis_client, verifier), server
    )
    server.add_insecure_port(f"{GRPC_BIND}:{GRPC_PORT}")

    logger.info("ingestion server v%s on :%d -> stream=%s", VERSION, GRPC_PORT, STREAM_NAME)
    await server.start()

    stop = asyncio.Event()

    def _request_stop(*_):
        logger.info("shutdown signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _request_stop)

    await stop.wait()
    # Drain in-flight RPCs before dying so a rolling deploy loses nothing.
    await server.stop(grace=10)
    await redis_client.aclose()
    await pool.close()
    logger.info("stopped. auth cache: %s", STATS)


if __name__ == "__main__":
    asyncio.run(serve())
