"""Tests for the SDK and the dataset/experiment API.

The SDK tests assert the two rules it promises, because both are the kind of
thing that only shows up in production if you do not test for it:

  1. it never breaks the app it observes
  2. it never changes what the app returns or raises
"""

from __future__ import annotations

import base64
import os
import sys
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    os.path.join(ROOT, "worker"),
    os.path.join(ROOT, "shared"),
    os.path.join(ROOT, "sdk"),
    os.path.join(ROOT, "api-server"),
]

DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://openweave:openweave@localhost:5432/openweave_test",
)
os.environ["DATABASE_URL"] = DB_URL.replace("+asyncpg", "")

import models as M  # noqa: E402
from auth import generate_key_pair, hash_secret  # noqa: E402
from openweave import OpenWeave  # noqa: E402

PROJECT = "test-project"
PK, SK = generate_key_pair()
AUTH = {"authorization": "Basic " + base64.b64encode(f"{PK}:{SK}".encode()).decode()}


# --------------------------------------------------------------------------- #
# SDK — no server required
# --------------------------------------------------------------------------- #
def _offline_client() -> tuple[OpenWeave, list]:
    """A client whose transport is replaced by a list, so we can read the wire."""
    client = OpenWeave(public_key="pk-ow-x", secret_key="sk-ow-x", enabled=False)
    captured: list = []
    client.enabled = True                       # re-enable tracing logic only
    client._enqueue = captured.append           # ...with no transport at all
    return client, captured


def _kind(event) -> str:
    return event.WhichOneof("body")


def test_decorators_nest_via_contextvars():
    """v1 minted a new trace per call, so nesting was impossible. This is the fix."""
    ow, events = _offline_client()

    @ow.observe(type="RETRIEVER")
    def search(q):
        return ["doc"]

    @ow.observe(type="GENERATION")
    def answer(q, docs):
        return "answer"

    @ow.observe(type="AGENT")
    def handle(q):
        return answer(q, search(q))

    with ow.trace(name="run", input="q") as trace:
        handle("q")

    creates = [e for e in events if _kind(e) == "observation_create"]
    by_name = {e.observation_create.name: e.observation_create for e in creates}
    assert set(by_name) == {"handle", "search", "answer"}

    # One trace for all three spans — not three traces.
    assert {o.trace_id for o in by_name.values()} == {trace.id}
    assert by_name["handle"].parent_observation_id == ""
    assert by_name["search"].parent_observation_id == by_name["handle"].id
    assert by_name["answer"].parent_observation_id == by_name["handle"].id


@pytest.mark.asyncio
async def test_async_functions_nest_too():
    ow, events = _offline_client()

    @ow.observe(type="TOOL")
    async def fetch(u):
        return "body"

    @ow.observe(type="AGENT")
    async def run(u):
        return await fetch(u)

    with ow.trace(name="async-run"):
        await run("http://x")

    creates = {e.observation_create.name: e.observation_create
               for e in events if _kind(e) == "observation_create"}
    assert creates["fetch"].parent_observation_id == creates["run"].id


def test_exception_is_captured_and_reraised_unchanged():
    """v1 emitted NOTHING when the wrapped function raised — the traces you most
    want were exactly the ones it dropped."""
    ow, events = _offline_client()

    @ow.observe(type="TOOL")
    def explode():
        raise ValueError("upstream timed out")

    with ow.trace(name="run"):
        with pytest.raises(ValueError, match="upstream timed out"):
            explode()

    updates = [e.observation_update for e in events if _kind(e) == "observation_update"]
    assert len(updates) == 1
    assert updates[0].level == 4          # OBSERVATION_LEVEL_ERROR
    assert "ValueError: upstream timed out" in updates[0].status_message
    assert updates[0].HasField("end_time")


def test_sdk_never_breaks_the_app_when_the_transport_is_down():
    """Point the SDK at a dead port; decorated code must behave identically."""
    ow = OpenWeave(public_key="pk-ow-x", secret_key="sk-ow-x",
                   host="127.0.0.1", port=1, flush_interval=0.05)

    @ow.observe(type="AGENT")
    def work(x):
        return x * 2

    @ow.observe(type="TOOL")
    def fails():
        raise RuntimeError("boom")

    assert work(21) == 42
    with pytest.raises(RuntimeError, match="boom"):
        fails()
    ow.flush(timeout=2)
    ow.shutdown()


def test_disabled_client_is_a_noop():
    ow = OpenWeave(public_key="", secret_key="")
    assert not ow.enabled

    @ow.observe()
    def work():
        return "fine"

    assert work() == "fine"
    assert ow.current().update(output="x") is not None   # never returns None


def test_full_queue_drops_instead_of_blocking():
    """Back-pressuring the caller to protect telemetry is the wrong trade."""
    ow = OpenWeave(public_key="pk-ow-x", secret_key="sk-ow-x",
                   host="127.0.0.1", port=1, max_queue=5, flush_interval=999)
    import openweave_pb2 as pb
    for _ in range(50):
        ow._enqueue(pb.Event(event_id=str(uuid.uuid4())))
    assert ow.stats["dropped"] > 0
    ow._stop.set()


def test_generation_records_model_and_usage():
    """v1 hardcoded model='unknown' and token_count=0 in its decorator."""
    ow, events = _offline_client()

    @ow.observe(type="GENERATION")
    def call(prompt):
        ow.current().first_token()
        ow.current().update(model="gpt-4o", usage={"input": 100, "output": 20})
        return "hi"

    with ow.trace(name="run"):
        call("hello")

    updates = [e.observation_update for e in events if _kind(e) == "observation_update"]
    enriched = next(u for u in updates if u.model)
    assert enriched.model == "gpt-4o"
    assert dict(enriched.usage_details) == {"input": 100, "output": 20}
    assert any(u.HasField("completion_start_time") for u in updates)


# --------------------------------------------------------------------------- #
# Dataset / experiment API
# --------------------------------------------------------------------------- #
SYNC_URL = DB_URL.replace("+asyncpg", "")


@pytest.fixture
def api():
    """Sync fixture on purpose.

    TestClient drives the ASGI app through its own event-loop portal; handing it
    a client created inside an async fixture deadlocks, because two loops end up
    waiting on each other. Schema setup here uses a synchronous engine.
    """
    from sqlalchemy import create_engine

    engine = create_engine(SYNC_URL)
    M.Base.metadata.drop_all(engine)
    M.Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(M.Project.__table__.insert().values(id=PROJECT, name="test"))
        conn.execute(M.ApiKey.__table__.insert().values(
            id="k1", project_id=PROJECT, public_key=PK,
            hashed_secret_key=hash_secret(SK), display_secret_key="sk-ow-...x",
        ))
    engine.dispose()

    from fastapi.testclient import TestClient
    import main

    with TestClient(main.app) as client:
        client.headers.update(AUTH)
        yield client


def _seed_run(client, dataset, run, item_id, trace_id, score=None, cost=None):
    """Link one item->trace for a run, and optionally attach a score and cost."""
    from sqlalchemy import create_engine

    client.post(f"/datasets/{dataset}/runs", json={"name": run})
    client.post(f"/datasets/{dataset}/runs/{run}/items",
                json={"dataset_item_id": item_id, "trace_id": trace_id})

    engine = create_engine(SYNC_URL)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO traces (project_id, id, timestamp, is_experiment,
                                created_at, updated_at)
            VALUES (:p, :t, now(), true, now(), now())
            ON CONFLICT DO NOTHING
        """), {"p": PROJECT, "t": trace_id})
        if cost is not None:
            conn.execute(text("""
                INSERT INTO observations (project_id, id, trace_id, type,
                    start_time, end_time, level, total_cost, created_at, updated_at)
                VALUES (:p, :o, :t, 'GENERATION', now(), now() + interval '1 s',
                        'DEFAULT', :c, now(), now())
            """), {"p": PROJECT, "o": str(uuid.uuid4()), "t": trace_id, "c": cost})
        if score is not None:
            conn.execute(text("""
                INSERT INTO scores (project_id, id, trace_id, name, data_type,
                    value, source, timestamp, created_at)
                VALUES (:p, :i, :t, 'relevance', 'NUMERIC', :v, 'EVAL', now(), now())
            """), {"p": PROJECT, "i": str(uuid.uuid4()), "t": trace_id, "v": score})
    engine.dispose()


def test_api_requires_auth(api):
    assert api.get("/traces", headers={"authorization": ""}).status_code == 401
    assert api.get("/traces").status_code == 200


def test_dataset_crud_and_from_trace(api):
    assert api.post("/datasets", json={"name": "qa"}).status_code == 201
    item = api.post("/datasets/qa/items",
                    json={"input": "q1", "expected_output": "a1"})
    assert item.status_code == 201
    assert len(api.get("/datasets/qa/items").json()["data"]) == 1
    assert api.get("/datasets").json()["data"][0]["item_count"] == 1
    # Unknown dataset is a 404, not a 500.
    assert api.post("/datasets/nope/items", json={"input": "x"}).status_code == 404


def test_compare_reports_score_and_cost_deltas(api):
    api.post("/datasets", json={"name": "qa"})
    items = [api.post("/datasets/qa/items",
                      json={"input": f"q{i}", "expected_output": f"a{i}"}).json()["id"]
             for i in range(3)]

    for i, item in enumerate(items):
        _seed_run(api, "qa", "v1", item, f"t-v1-{i}", score=0.5, cost=0.001)
    for i, item in enumerate(items):
        _seed_run(api, "qa", "v2", item, f"t-v2-{i}", score=0.9, cost=0.002)

    report = api.get("/datasets/qa/compare", params={"runs": "v1,v2"}).json()
    by_run = {r["run"]: r for r in report["runs"]}
    assert by_run["v1"]["scores"]["relevance"]["mean"] == pytest.approx(0.5)
    assert by_run["v2"]["scores"]["relevance"]["mean"] == pytest.approx(0.9)
    assert by_run["v1"]["items"] == 3

    delta = report["deltas"]["v2 vs v1"]
    assert delta["relevance"] == pytest.approx(0.4)
    assert delta["avg_cost_pct"] == pytest.approx(100.0)

    # Per-item rows exist so a flat mean cannot hide a per-case regression.
    assert len(report["items"]) == 3
    assert set(report["items"][0]["runs"]) == {"v1", "v2"}


def test_compare_needs_two_runs(api):
    api.post("/datasets", json={"name": "qa"})
    assert api.get("/datasets/qa/compare",
                   params={"runs": "only-one"}).status_code == 400


def test_run_item_link_is_idempotent(api):
    api.post("/datasets", json={"name": "qa"})
    item = api.post("/datasets/qa/items", json={"input": "q"}).json()["id"]
    api.post("/datasets/qa/runs", json={"name": "v1"})
    first = api.post("/datasets/qa/runs/v1/items",
                     json={"dataset_item_id": item, "trace_id": "t1"})
    second = api.post("/datasets/qa/runs/v1/items",
                      json={"dataset_item_id": item, "trace_id": "t2"})
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] == second.json()["id"], "re-running must not duplicate"


def test_trace_tree_keeps_a_span_whose_parent_is_missing(api):
    """Regression: the tree came from a recursive CTE walking down from root
    spans, so a span whose parent had not arrived (out-of-order delivery, or a
    dead-lettered parent) vanished from the API together with its subtree."""
    from sqlalchemy import create_engine

    engine = create_engine(SYNC_URL)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO traces (project_id, id, timestamp, is_experiment,
                                created_at, updated_at)
            VALUES (:p, 't1', now(), false, now(), now())
        """), {"p": PROJECT})
        for oid, parent, offset in (("root", None, 0), ("child", "root", 1),
                                    ("orphan", "never-arrived", 2),
                                    ("orphan-child", "orphan", 3)):
            conn.execute(text("""
                INSERT INTO observations (project_id, id, trace_id,
                    parent_observation_id, type, start_time, level,
                    created_at, updated_at)
                VALUES (:p, :o, 't1', :parent, 'SPAN',
                        now() + make_interval(secs => :off), 'DEFAULT', now(), now())
            """), {"p": PROJECT, "o": oid, "parent": parent, "off": offset})
    engine.dispose()

    trace = api.get("/traces/t1").json()
    assert trace["span_count"] == 4
    roots = {n["id"]: n for n in trace["observations"]}
    assert set(roots) == {"root", "orphan"}
    assert [c["id"] for c in roots["root"]["children"]] == ["child"]
    assert [c["id"] for c in roots["orphan"]["children"]] == ["orphan-child"]
    assert roots["orphan"]["parent_missing"] is True
    assert roots["root"]["parent_missing"] is False


def test_compare_counts_items_that_have_no_score_yet(api):
    """Regression: totals were grouped by score name, so an unscored item (still
    settling, or a judge ERROR) formed its own group and overwrote the run's
    item count and average cost."""
    api.post("/datasets", json={"name": "qa"})
    items = [api.post("/datasets/qa/items", json={"input": f"q{i}"}).json()["id"]
             for i in range(3)]
    for i, (score, cost) in enumerate(((0.5, 0.001), (0.5, 0.001), (None, 0.004))):
        _seed_run(api, "qa", "v1", items[i], f"t-v1-{i}", score=score, cost=cost)
    for i in range(3):
        _seed_run(api, "qa", "v2", items[i], f"t-v2-{i}", score=0.9, cost=0.002)

    report = api.get("/datasets/qa/compare", params={"runs": "v1,v2"}).json()
    v1 = next(r for r in report["runs"] if r["run"] == "v1")
    assert v1["items"] == 3
    assert v1["avg_cost"] == pytest.approx(0.002)
    assert v1["scores"]["relevance"]["n"] == 2
    assert v1["scores"]["relevance"]["mean"] == pytest.approx(0.5)
    assert report["deltas"]["v2 vs v1"]["avg_cost_pct"] == pytest.approx(0.0)


def test_compare_includes_scores_anchored_to_the_run_item(api):
    """A score may anchor to dataset_run_item_id instead of trace_id, as the
    SDK's score(dataset_run_item_id=...) does; compare used to ignore those."""
    from sqlalchemy import create_engine

    api.post("/datasets", json={"name": "qa"})
    item = api.post("/datasets/qa/items", json={"input": "q"}).json()["id"]
    _seed_run(api, "qa", "v1", item, "t-v1", cost=0.001)
    _seed_run(api, "qa", "v2", item, "t-v2", score=0.9, cost=0.001)
    run_item_id = api.post("/datasets/qa/runs/v1/items",
                           json={"dataset_item_id": item, "trace_id": "t-v1"}).json()["id"]
    engine = create_engine(SYNC_URL)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO scores (project_id, id, dataset_run_item_id, name, data_type,
                value, source, timestamp, created_at)
            VALUES (:p, 's-run-item', :ri, 'relevance', 'NUMERIC', 0.6, 'API',
                    now(), now())
        """), {"p": PROJECT, "ri": run_item_id})
    engine.dispose()

    report = api.get("/datasets/qa/compare", params={"runs": "v1,v2"}).json()
    v1 = next(r for r in report["runs"] if r["run"] == "v1")
    assert v1["scores"]["relevance"]["mean"] == pytest.approx(0.6)
    assert report["items"][0]["runs"]["v1"]["scores"]["relevance"] == pytest.approx(0.6)
