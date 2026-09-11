"""Tests for the cost engine and the evaluation loop.

The eval tests inject a fake judge, so the whole path — rules, sampling,
filters, variable mapping, prompt rendering, schema validation, score and job
rows — is exercised without an LLM. That is deliberate: an eval pipeline you
can only test by calling a model is an eval pipeline you will not test.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    os.path.join(ROOT, "worker"),
    os.path.join(ROOT, "shared"),
    os.path.join(ROOT, "evaluator"),
]

import eval_worker as EV  # noqa: E402
import models as M  # noqa: E402
import worker as W  # noqa: E402
from judge import JudgeError, render_prompt, validate_output  # noqa: E402
from pricing import ModelPriceCache, compute_cost, resolve_costs  # noqa: E402

DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://openweave:openweave@localhost:5432/openweave_test",
)
T0 = datetime(2026, 9, 11, 12, 0, 0, tzinfo=timezone.utc)
PROJECT = "test-project"


def event(kind, body, ts=None):
    return {"event_id": str(uuid.uuid4()), "project_id": PROJECT, "type": kind,
            "ts": (ts or T0).isoformat(), "body": body}


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine(DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(M.Base.metadata.drop_all)
        await conn.run_sync(M.Base.metadata.create_all)
        await conn.execute(M.Project.__table__.insert().values(id=PROJECT, name="test"))
        # gpt-4o at $2.50 / $10.00 per million, plus a cheaper older sheet.
        await conn.execute(text("""
            INSERT INTO models (id, project_id, model_name, match_pattern, start_date)
            VALUES ('m-new', NULL, 'gpt-4o', '(?i)^gpt-4o(-\\d{4}-\\d{2}-\\d{2})?$',
                    '2026-06-01T00:00:00Z'),
                   ('m-old', NULL, 'gpt-4o', '(?i)^gpt-4o(-\\d{4}-\\d{2}-\\d{2})?$',
                    '2025-01-01T00:00:00Z')
        """))
        await conn.execute(text("""
            INSERT INTO prices (id, model_id, usage_type, price) VALUES
              ('p1','m-new','input',  0.0000025),
              ('p2','m-new','output', 0.0000100),
              ('p3','m-new','cache_read', 0.00000125),
              ('p4','m-old','input',  0.0000050),
              ('p5','m-old','output', 0.0000200)
        """))
    yield engine
    await engine.dispose()


class FakeRedis:
    def __init__(self):
        self.kv, self.hashes, self.streams = {}, {}, {}

    async def set(self, k, v, nx=False, ex=None):
        if nx and k in self.kv:
            return None
        self.kv[k] = v
        return True

    async def hincrby(self, h, k, n):
        self.hashes.setdefault(h, {})
        self.hashes[h][k] = self.hashes[h].get(k, 0) + n
        return self.hashes[h][k]

    async def hdel(self, h, *ks):
        for k in ks:
            self.hashes.get(h, {}).pop(k, None)


# --------------------------------------------------------------------------- #
# Cost engine
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_cost_resolved_from_usage_and_price_table(db):
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("observation_create", {
                "id": "g1", "trace_id": "t1", "type": "OBSERVATION_TYPE_GENERATION",
                "model": "gpt-4o-2026-08-01", "start_time": T0.isoformat(),
            }),
            event("observation_update", {
                "id": "g1", "trace_id": "t1",
                "usage_details": {"input": "1000000", "output": "500000"},
            }),
        ])
        await resolve_costs(conn, ModelPriceCache(), {(PROJECT, "g1")})

    async with db.connect() as conn:
        row = (await conn.execute(text(
            "SELECT total_cost, cost_details->>'input' ci,"
            " cost_details->>'output' co, internal_model_id m"
            " FROM observations WHERE id='g1'"))).one()
    # 1M input @ $2.50 + 0.5M output @ $10.00 = 2.50 + 5.00
    assert float(row.total_cost) == pytest.approx(7.50)
    assert float(row.ci) == pytest.approx(2.50)
    assert float(row.co) == pytest.approx(5.00)
    assert row.m == "m-new", "the dated model snapshot must match the regex"


@pytest.mark.asyncio
async def test_price_is_the_one_in_force_at_the_observation_time(db):
    """A trace from before the price change must keep the old price."""
    old_time = datetime(2025, 6, 1, tzinfo=timezone.utc)
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("observation_create", {
                "id": "g_old", "trace_id": "t1", "type": "OBSERVATION_TYPE_GENERATION",
                "model": "gpt-4o", "start_time": old_time.isoformat(),
            }, ts=old_time),
            event("observation_update", {
                "id": "g_old", "trace_id": "t1",
                "usage_details": {"input": "1000000"},
            }, ts=old_time),
        ])
        await resolve_costs(conn, ModelPriceCache(), {(PROJECT, "g_old")})

    async with db.connect() as conn:
        row = (await conn.execute(text(
            "SELECT total_cost, internal_model_id m FROM observations WHERE id='g_old'"
        ))).one()
    assert row.m == "m-old"
    assert float(row.total_cost) == pytest.approx(5.00), "old sheet: $5.00/M input"


@pytest.mark.asyncio
async def test_client_supplied_cost_is_never_overwritten(db):
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("observation_create", {
                "id": "g2", "trace_id": "t1", "type": "OBSERVATION_TYPE_GENERATION",
                "model": "gpt-4o", "start_time": T0.isoformat(),
                "usage_details": {"input": "1000000"},
                "cost_details": {"input": 0.99},
            }),
        ])
        await resolve_costs(conn, ModelPriceCache(), {(PROJECT, "g2")})

    async with db.connect() as conn:
        row = (await conn.execute(text(
            "SELECT total_cost, provided_cost_details->>'input' pi"
            " FROM observations WHERE id='g2'"))).one()
    assert float(row.pi) == pytest.approx(0.99)
    assert float(row.total_cost) == pytest.approx(0.99)


@pytest.mark.asyncio
async def test_unknown_model_stays_unpriced_rather_than_zero(db):
    async with db.begin() as conn:
        await W.apply_events(conn, [
            event("observation_create", {
                "id": "g3", "trace_id": "t1", "type": "OBSERVATION_TYPE_GENERATION",
                "model": "some-model-nobody-seeded", "start_time": T0.isoformat(),
                "usage_details": {"input": "1000"},
            }),
        ])
        await resolve_costs(conn, ModelPriceCache(), {(PROJECT, "g3")})

    async with db.connect() as conn:
        cost = (await conn.execute(text(
            "SELECT total_cost FROM observations WHERE id='g3'"))).scalar()
    assert cost is None, "unknown price must read as unknown, not as free"


def test_aggregate_usage_keys_are_not_double_counted():
    from pricing import ModelPrice
    import re
    price = ModelPrice("m", "m", None, re.compile("."), None,
                       {"input": Decimal("0.001"), "output": Decimal("0.002"),
                        "total": Decimal("0.009")})
    details, total = compute_cost(
        {"input": 100, "output": 50, "total": 150}, price)
    assert "total" not in details
    assert total == Decimal("0.2")


# --------------------------------------------------------------------------- #
# Judge helpers
# --------------------------------------------------------------------------- #
def test_render_prompt_leaves_literal_json_alone():
    out = render_prompt('Return {"score": 0.5} for {{name}}', {"name": "x"})
    assert out == 'Return {"score": 0.5} for x'


def test_validate_output_rejects_out_of_range_and_wrong_types():
    schema = {"type": "object", "required": ["score"],
              "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}}}
    assert validate_output({"score": 0.8}, schema)["score"] == 0.8
    for bad in ({"reason": "no score"}, {"score": "high"}, {"score": 1.5}):
        with pytest.raises(JudgeError):
            validate_output(bad, schema)


def test_sampling_is_deterministic():
    """Re-announcing a trace must not give it another roll of the dice."""
    values = {EV.sample_hash("trace-abc") for _ in range(10)}
    assert len(values) == 1
    sampled = [t for t in (f"t{i}" for i in range(2000))
               if EV.sample_hash(t) < 0.1]
    assert 150 < len(sampled) < 250, "10% sampling should land near 10%"


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #
async def _seed_evaluator(conn, sampling=1.0, filter_=None, schema=None):
    await conn.execute(text("""
        INSERT INTO evaluators (project_id, id, name, created_at)
        VALUES (:p, 'ev1', 'relevance-judge', now())
    """), {"p": PROJECT})
    await conn.execute(text("""
        INSERT INTO evaluator_versions
            (id, project_id, evaluator_id, version, prompt, model, provider,
             model_params, variable_mapping, output_schema, score_name,
             score_data_type, created_at)
        VALUES ('v1', :p, 'ev1', 1,
                'Q: {{input}}\nA: {{output}}', 'llama3', 'ollama',
                '{}', '{"input":"trace.input","output":"root.output"}',
                CAST(:schema AS jsonb), 'relevance', 'NUMERIC', now())
    """), {"p": PROJECT, "schema": schema or
           '{"type":"object","required":["score"],'
           '"properties":{"score":{"type":"number","minimum":0,"maximum":1}}}'})
    await conn.execute(text("""
        INSERT INTO evaluation_rules
            (project_id, id, evaluator_id, target, filter, sampling_rate,
             is_active, created_at)
        VALUES (:p, 'r1', 'ev1', 'TRACE', CAST(:f AS jsonb), :s, true, now())
    """), {"p": PROJECT, "f": filter_, "s": sampling})


async def _seed_trace(conn, trace_id="t1", name="support-agent"):
    await W.apply_events(conn, [
        event("trace_create", {"id": trace_id, "name": name,
                               "input": "How do I reset my password?",
                               "timestamp": T0.isoformat()}),
        event("observation_create", {
            "id": f"{trace_id}-root", "trace_id": trace_id,
            "type": "OBSERVATION_TYPE_AGENT", "name": "handle",
            "start_time": T0.isoformat(),
            "output": "Click 'forgot password' on the sign-in page.",
        }),
    ])


@pytest.mark.asyncio
async def test_eval_writes_score_and_completed_job(db):
    seen = {}

    async def fake_judge(provider, model, prompt, params):
        seen["prompt"] = prompt
        return {"score": 0.82, "reason": "directly answers the question"}

    async with db.begin() as conn:
        await _seed_evaluator(conn)
        await _seed_trace(conn)
        outcomes = await EV.evaluate_trace(
            conn, FakeRedis(), PROJECT, "t1", judge_fn=fake_judge)

    assert outcomes and "scored relevance=0.82" in outcomes[0]
    # variable_mapping resolved trace.input and root.output
    assert "How do I reset my password?" in seen["prompt"]
    assert "forgot password" in seen["prompt"]

    async with db.connect() as conn:
        score = (await conn.execute(text(
            "SELECT name, value, source::text src, comment, evaluator_version_id v"
            " FROM scores"))).one()
        job = (await conn.execute(text(
            "SELECT status::text st, error, judge_latency_ms FROM job_executions"
        ))).one()
    assert (score.name, score.src, score.v) == ("relevance", "EVAL", "v1")
    assert score.value == pytest.approx(0.82)
    assert score.comment == "directly answers the question"
    assert job.st == "COMPLETED" and job.error is None
    assert job.judge_latency_ms is not None


@pytest.mark.asyncio
async def test_judge_failure_is_recorded_not_swallowed(db):
    async def broken_judge(*_a, **_kw):
        raise JudgeError("ollama connection refused")

    async with db.begin() as conn:
        await _seed_evaluator(conn)
        await _seed_trace(conn)
        outcomes = await EV.evaluate_trace(
            conn, FakeRedis(), PROJECT, "t1", judge_fn=broken_judge)

    assert "error" in outcomes[0]
    async with db.connect() as conn:
        job = (await conn.execute(text(
            "SELECT status::text st, error FROM job_executions"))).one()
        n = (await conn.execute(text("SELECT count(*) FROM scores"))).scalar()
    assert job.st == "ERROR"
    assert "connection refused" in job.error
    assert n == 0, "a failed judge must not produce a score"


@pytest.mark.asyncio
async def test_out_of_range_verdict_is_an_error_not_a_silent_score(db):
    async def lying_judge(*_a, **_kw):
        return {"score": 42, "reason": "very good"}

    async with db.begin() as conn:
        await _seed_evaluator(conn)
        await _seed_trace(conn)
        await EV.evaluate_trace(conn, FakeRedis(), PROJECT, "t1", judge_fn=lying_judge)

    async with db.connect() as conn:
        job = (await conn.execute(text(
            "SELECT status::text st, error FROM job_executions"))).one()
        n = (await conn.execute(text("SELECT count(*) FROM scores"))).scalar()
    assert job.st == "ERROR" and "maximum" in job.error
    assert n == 0


@pytest.mark.asyncio
async def test_rule_filter_and_dedupe(db):
    async def judge(*_a, **_kw):
        return {"score": 1.0, "reason": "ok"}

    async with db.begin() as conn:
        await _seed_evaluator(
            conn, filter_='[{"column":"name","op":"=","value":"support-agent"}]')
        await _seed_trace(conn, "t1", name="support-agent")
        await _seed_trace(conn, "t2", name="billing-agent")
        redis_client = FakeRedis()

        assert await EV.evaluate_trace(conn, redis_client, PROJECT, "t1", judge_fn=judge)
        assert not await EV.evaluate_trace(conn, redis_client, PROJECT, "t2",
                                           judge_fn=judge), "filtered out"
        # Second announcement of the same trace must not score it twice.
        assert not await EV.evaluate_trace(conn, redis_client, PROJECT, "t1",
                                           judge_fn=judge)

    async with db.connect() as conn:
        n = (await conn.execute(text("SELECT count(*) FROM scores"))).scalar()
    assert n == 1


@pytest.mark.asyncio
async def test_settle_gate(db):
    async with db.begin() as conn:
        await _seed_trace(conn)
        assert not await EV.is_settled(conn, PROJECT, "t1"), "just written"
        await conn.execute(text(
            "UPDATE observations SET updated_at = now() - interval '1 hour'"))
        await conn.execute(text(
            "UPDATE traces SET updated_at = now() - interval '1 hour'"))
        assert await EV.is_settled(conn, PROJECT, "t1")


class FakeDueRedis(FakeRedis):
    """FakeRedis plus the sorted-set and ack calls the eval loop makes."""

    def __init__(self):
        super().__init__()
        self.zsets = {}

    async def zadd(self, key, mapping, gt=False, xx=False):
        zset = self.zsets.setdefault(key, {})
        for member, score in mapping.items():
            if member in zset:
                if gt and score <= zset[member]:
                    continue
            elif xx:
                continue
            zset[member] = score

    async def zrangebyscore(self, key, _low, high, start=0, num=None):
        due = sorted((s, m) for m, s in self.zsets.get(key, {}).items() if s <= high)
        return [m for _, m in due][start:(start + num) if num else None]

    async def zrem(self, key, *members):
        for member in members:
            self.zsets.get(key, {}).pop(member, None)

    async def xack(self, *_args):
        return 1


@pytest.mark.asyncio
async def test_announced_trace_is_judged_only_after_it_goes_quiet(db, monkeypatch):
    """Regression: unsettled traces were re-XADDed and re-read within
    milliseconds, so the bounce cap ran out and half-written traces were judged
    and then never judged again. The judge must see an answer that lands after
    the announcement."""
    monkeypatch.setattr(EV, "SETTLE_SECONDS", 0.5)
    prompts = []

    async def judge(_provider, _model, prompt, _params):
        prompts.append(prompt)
        return {"score": 1.0, "reason": "ok"}

    async with db.begin() as conn:
        await _seed_evaluator(conn)
        await W.apply_events(conn, [
            event("trace_create", {"id": "t1", "name": "support-agent",
                                   "input": "How do I reset my password?"}),
            event("observation_create", {"id": "root", "trace_id": "t1",
                                         "type": "OBSERVATION_TYPE_AGENT",
                                         "start_time": T0.isoformat()}),
        ])
    redis_client = FakeDueRedis()
    await EV.handle_message(redis_client, "1-0", {"project_id": PROJECT, "trace_id": "t1"})
    assert await EV.drain_due(db, redis_client, judge_fn=judge) == 0, "not due yet"

    await asyncio.sleep(0.4)
    async with db.begin() as conn:  # the answer lands just before the check is due
        await W.apply_events(conn, [
            event("observation_update", {"id": "root", "trace_id": "t1",
                                         "output": "Click 'forgot password'."}),
        ])
    await asyncio.sleep(0.2)
    await EV.drain_due(db, redis_client, judge_fn=judge)
    assert prompts == [], "last write was 0.2s ago, so the trace has not settled"

    await asyncio.sleep(0.7)
    await EV.drain_due(db, redis_client, judge_fn=judge)
    assert len(prompts) == 1
    assert "forgot password" in prompts[0]
