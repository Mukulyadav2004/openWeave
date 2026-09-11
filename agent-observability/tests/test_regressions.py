"""Regression tests for the four bugs found in end-to-end testing.

Each test fails on the pre-fix code and passes after. Keep them: they are the
cheapest proof that the ingestion path is correct, and they are the start of
the test suite that closes gap 7.

    pytest tests/ -v

Needs a Postgres and a Redis. Point at them with:
    TEST_DATABASE_URL=postgresql+asyncpg://...  TEST_REDIS_URL=redis://...
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    os.path.join(ROOT, "worker"),
    os.path.join(ROOT, "shared"),
    os.path.join(ROOT, "ingestion-server"),
]

import models as M  # noqa: E402
import worker as W  # noqa: E402
from auth import ApiKeyVerifier, AuthError, generate_key_pair, hash_secret  # noqa: E402

DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://openweave:openweave@localhost:5432/openweave_test",
)

T0 = datetime(2026, 9, 11, 12, 0, 0, tzinfo=timezone.utc)
PROJECT = "test-project"


def event(kind: str, body: dict, ts: datetime | None = None) -> dict:
    """A stream payload shaped exactly like the one the ingestion server writes."""
    return {
        "event_id": str(uuid.uuid4()),
        "project_id": PROJECT,
        "type": kind,
        "ts": (ts or T0).isoformat(),
        "body": body,
    }


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine(DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(M.Base.metadata.drop_all)
        await conn.run_sync(M.Base.metadata.create_all)
        await conn.execute(
            M.Project.__table__.insert().values(id=PROJECT, name="test")
        )
    yield engine
    await engine.dispose()


# --------------------------------------------------------------------------- #
# Bug 1 — an observation_update carrying end_time was dead-lettered.
#
# start_time defaulted to the EVENT timestamp, which is stamped just after the
# span closed and is therefore LATER than end_time. Postgres evaluates
# ck_obs_end_after_start against the proposed INSERT row even when ON CONFLICT
# turns the statement into an UPDATE, so every span close was rejected.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_update_with_end_time_is_not_rejected(db):
    end = T0 + timedelta(seconds=3)
    # The event is stamped AFTER the span ended — the realistic ordering.
    stamped_after = end + timedelta(milliseconds=5)

    async with db.begin() as conn:
        await W.apply_events(
            conn,
            [event("observation_update",
                   {"id": "o1", "trace_id": "t1", "end_time": end.isoformat()},
                   ts=stamped_after)],
        )

    async with db.connect() as conn:
        row = (await conn.execute(text(
            "SELECT start_time, end_time FROM observations WHERE id='o1'"
        ))).one()
    assert row.end_time == end
    assert row.start_time <= row.end_time, "start_time must not exceed end_time"


@pytest.mark.asyncio
async def test_later_create_supplies_the_real_start_time(db):
    """The end_time placeholder must be replaced once the create arrives."""
    end = T0 + timedelta(seconds=3)
    real_start = T0

    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("observation_update",
                  {"id": "o1", "trace_id": "t1", "end_time": end.isoformat(),
                   "usage_details": {"input": "100", "output": "50"}},
                  ts=end),
        ])
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("observation_create",
                  {"id": "o1", "trace_id": "t1", "type": "OBSERVATION_TYPE_GENERATION",
                   "name": "answer", "start_time": real_start.isoformat()}),
        ])

    async with db.connect() as conn:
        row = (await conn.execute(text(
            "SELECT start_time, end_time, type::text ty, latency_ms,"
            " usage_details->>'input' inp FROM observations WHERE id='o1'"
        ))).one()
    assert row.start_time == real_start
    assert row.ty == "GENERATION"
    assert row.latency_ms == 3000.0
    assert row.inp == "100", "the create must not wipe usage the update stored"


# --------------------------------------------------------------------------- #
# Bug 2 — stub traces overwrote real traces.
#
# Stubs were merge-upserted, and their placeholder timestamp / is_experiment
# are non-NULL, so COALESCE picked them over the real values.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stub_trace_does_not_overwrite_a_real_trace(db):
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("trace_create", {
                "id": "t1", "name": "experiment-run", "timestamp": T0.isoformat(),
                "is_experiment": True,
            }),
        ])
    # A later batch references t1 from an observation, which triggers a stub.
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("observation_create", {
                "id": "o9", "trace_id": "t1", "type": "OBSERVATION_TYPE_SPAN",
                "start_time": T0.isoformat(),
            }),
        ])

    async with db.connect() as conn:
        row = (await conn.execute(text(
            "SELECT name, timestamp, is_experiment FROM traces WHERE id='t1'"
        ))).one()
    assert row.name == "experiment-run"
    assert row.timestamp == T0, "stub's now() must not replace the real timestamp"
    assert row.is_experiment is True, "stub's False must not clear is_experiment"


# --------------------------------------------------------------------------- #
# Bug 3 — merge upserts wrote placeholders into columns the event never sent.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_partial_trace_create_does_not_reset_unsent_columns(db):
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("trace_create", {
                "id": "t1", "name": "run", "timestamp": T0.isoformat(),
                "is_experiment": True, "release": "v1.4",
            }),
        ])
    # A second trace_create carrying only output — everything else is unsent.
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("trace_create", {"id": "t1", "output": "done"},
                  ts=T0 + timedelta(minutes=5)),
        ])

    async with db.connect() as conn:
        row = (await conn.execute(text(
            "SELECT output, timestamp, is_experiment, release FROM traces WHERE id='t1'"
        ))).one()
    assert row.output == "done"
    assert row.timestamp == T0
    assert row.is_experiment is True
    assert row.release == "v1.4"


@pytest.mark.asyncio
async def test_observation_create_does_not_reset_level_to_default(db):
    """An ERROR level set by an update must survive a later partial create."""
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("observation_update", {
                "id": "o1", "trace_id": "t1", "level": "OBSERVATION_LEVEL_ERROR",
                "status_message": "TimeoutError: upstream timed out",
            }),
        ])
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("observation_create", {
                "id": "o1", "trace_id": "t1", "type": "OBSERVATION_TYPE_TOOL",
                "name": "fetch", "start_time": T0.isoformat(),
            }),
        ])

    async with db.connect() as conn:
        row = (await conn.execute(text(
            "SELECT level::text lvl, status_message, type::text ty"
            " FROM observations WHERE id='o1'"
        ))).one()
    assert row.ty == "TOOL"
    assert row.lvl == "ERROR", "the create's DEFAULT placeholder must not win"
    assert "TimeoutError" in row.status_message


@pytest.mark.asyncio
async def test_batch_with_mixed_column_shapes(db):
    """_group_by_sent must handle several SET shapes in one executemany batch."""
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("trace_create", {"id": "a", "name": "one", "timestamp": T0.isoformat()}),
            event("trace_create", {"id": "b", "user_id": "u1"}),
            event("trace_create", {"id": "c", "name": "three", "release": "v2",
                                   "tags": ["x"]}),
        ])
    async with db.connect() as conn:
        n = (await conn.execute(text("SELECT count(*) FROM traces"))).scalar()
    assert n == 3


# --------------------------------------------------------------------------- #
# Bug 4 — a cache hit skipped the public_key and expiry checks.
# --------------------------------------------------------------------------- #
class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, k):
        return self.store.get(k)

    async def set(self, k, v, ex=None):
        self.store[k] = v

    async def delete(self, k):
        self.store.pop(k, None)


class FakePool:
    """Stands in for asyncpg: one key row, matched on (hash, public_key)."""

    def __init__(self, rows):
        self.rows = rows
        self.queries = 0

    async def fetchrow(self, _sql, key_hash, public_key):
        self.queries += 1
        for r in self.rows:
            if r["hashed_secret_key"] == key_hash and r["public_key"] == public_key:
                return r
        return None

    async def execute(self, *_a, **_kw):
        return None


def _header(pk: str, sk: str) -> str:
    import base64
    return "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode()


@pytest.mark.asyncio
async def test_cache_hit_still_checks_public_key():
    pk, sk = generate_key_pair()
    other_pk, _ = generate_key_pair()
    pool = FakePool([{
        "id": "k1", "project_id": "p1", "public_key": pk,
        "hashed_secret_key": hash_secret(sk), "expires_at": None,
    }])
    verifier = ApiKeyVerifier(pool, FakeRedis())

    assert (await verifier.verify(_header(pk, sk))).project_id == "p1"  # populates cache

    # Same secret, different public key. Pre-fix this rode the cache entry and
    # was accepted; it must now fall through to Postgres and be rejected.
    with pytest.raises(AuthError):
        await verifier.verify(_header(other_pk, sk))


@pytest.mark.asyncio
async def test_cache_hit_still_checks_expiry():
    pk, sk = generate_key_pair()
    row = {
        "id": "k1", "project_id": "p1", "public_key": pk,
        "hashed_secret_key": hash_secret(sk),
        "expires_at": datetime.now(timezone.utc) + timedelta(milliseconds=50),
    }
    verifier = ApiKeyVerifier(FakePool([row]), FakeRedis())

    assert (await verifier.verify(_header(pk, sk))).project_id == "p1"

    import asyncio
    await asyncio.sleep(0.1)
    # The cache entry is still live, but the key is not.
    with pytest.raises(AuthError, match="expired"):
        await verifier.verify(_header(pk, sk))


@pytest.mark.asyncio
async def test_cache_actually_serves_hits():
    """The fix must not accidentally disable caching."""
    pk, sk = generate_key_pair()
    pool = FakePool([{
        "id": "k1", "project_id": "p1", "public_key": pk,
        "hashed_secret_key": hash_secret(sk), "expires_at": None,
    }])
    verifier = ApiKeyVerifier(pool, FakeRedis())

    for _ in range(5):
        await verifier.verify(_header(pk, sk))
    assert pool.queries == 1, "only the first call should reach Postgres"
