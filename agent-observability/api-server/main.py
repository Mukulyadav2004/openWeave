"""Read + dataset API — v2.

Replaces the v1 FastAPI server, which is incompatible with the new schema and
had three problems beyond that: no authentication at all, `allow_origins=["*"]`
together with `allow_credentials=True`, and an endpoint that ran the LLM judge
synchronously inside the request by importing a "Lambda" handler from a sibling
directory via sys.path.

Everything here is project-scoped from the API key. No endpoint takes a
project_id, so there is no code path that can read across tenants.

The endpoint that matters is:

    GET /datasets/{name}/compare?runs=prompt-v6,prompt-v7

which is the whole point of the project: change something, re-run a fixed
dataset, and see whether quality went up or down and what it cost.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import asyncpg
import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from auth import STATS, ApiKeyVerifier, AuthContext, AuthError  # noqa: E402

VERSION = "2.0.0"
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://openweave:openweave@localhost:5432/openweave"
).replace("+asyncpg", "").replace("+psycopg2", "")
# REDIS_URL carries the password when the host requires one (Railway's does).
REDIS_URL = os.getenv("REDIS_URL") or "redis://{}:{}".format(
    os.getenv("REDIS_HOST", "localhost"), os.getenv("REDIS_PORT", "6379"))
# When set, a request that sends no credentials may READ this one project, so a
# hosted demo can be browsed without handing out a key. Writes always need one.
PUBLIC_DEMO_PROJECT = os.getenv("PUBLIC_DEMO_PROJECT", "")
UI_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "index.html")
# Explicit origins only. "*" with credentials is rejected by browsers anyway and
# reads as carelessness in review.
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "").split(",") if o]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=2, max_size=10)
    app.state.redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    app.state.verifier = ApiKeyVerifier(app.state.pool, app.state.redis)
    app.state.demo_project_id = None
    try:
        yield
    finally:
        await app.state.pool.close()
        await app.state.redis.aclose()


app = FastAPI(title="OpenWeave API", version=VERSION, lifespan=lifespan)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
        allow_methods=["GET", "POST"], allow_headers=["authorization", "content-type"],
    )


async def write_auth(request: Request) -> AuthContext:
    """Every write needs a real key, whether or not a public demo is enabled."""
    try:
        return await request.app.state.verifier.verify(
            request.headers.get("authorization")
        )
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from None


async def _demo_project_id(app) -> str | None:
    # Looked up on first use rather than at startup: the demo project is usually
    # created after the API first boots against a freshly migrated database.
    if app.state.demo_project_id is None:
        app.state.demo_project_id = await app.state.pool.fetchval(
            "SELECT id FROM projects WHERE name = $1", PUBLIC_DEMO_PROJECT)
    return app.state.demo_project_id


async def auth(request: Request) -> AuthContext:
    """Reads. With PUBLIC_DEMO_PROJECT set, a request that sends no credentials
    reads that one project. Credentials that ARE sent are verified as usual and
    never fall back to the demo, so a wrong key is still a 401."""
    if PUBLIC_DEMO_PROJECT and not request.headers.get("authorization"):
        project_id = await _demo_project_id(request.app)
        if project_id is not None:
            return AuthContext(project_id=project_id, api_key_id="public-demo")
    return await write_auth(request)


def _json(value):
    """asyncpg hands back jsonb as a string; decode for the response model."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health(request: Request):
    ok = True
    try:
        await request.app.state.pool.fetchval("SELECT 1")
    except Exception:  # noqa: BLE001
        ok = False
    return {"ok": ok, "version": VERSION, "public_demo": bool(PUBLIC_DEMO_PROJECT),
            "auth_cache": STATS}


@app.get("/", include_in_schema=False)
async def ui():
    """The single-file UI, served from the API's own origin so it needs no CORS."""
    if not os.path.exists(UI_INDEX):
        raise HTTPException(status_code=404, detail="ui/index.html not found")
    return FileResponse(UI_INDEX)


# --------------------------------------------------------------------------- #
# Traces
# --------------------------------------------------------------------------- #
@app.get("/traces")
async def list_traces(
    request: Request,
    ctx: AuthContext = Depends(auth),
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    release: str | None = None,
    is_experiment: bool | None = None,
    since: datetime | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Trace list with cost, latency and scores rolled up from observations.

    The rollup is computed here rather than denormalised onto `traces`. See
    docs/M1-DESIGN-NOTES.md for when that trade stops being the right one.
    """
    conditions = ["t.project_id = $1"]
    params: list[Any] = [ctx.project_id]

    for value, sql in ((name, "t.name = "), (user_id, "t.user_id = "),
                       (session_id, "t.session_id = "), (release, "t.release = ")):
        if value is not None:
            params.append(value)
            conditions.append(f"{sql}${len(params)}")
    if is_experiment is not None:
        params.append(is_experiment)
        conditions.append(f"t.is_experiment = ${len(params)}")
    if since is not None:
        params.append(since)
        conditions.append(f"t.timestamp >= ${len(params)}")

    params.extend([limit, offset])
    rows = await request.app.state.pool.fetch(f"""
        SELECT t.id, t.name, t.user_id, t.session_id, t.timestamp, t.release,
               t.version, t.tags, t.is_experiment, t.input, t.output,
               o.span_count, o.total_cost, o.latency_ms, s.scores
          FROM traces t
          LEFT JOIN LATERAL (
                SELECT count(*) AS span_count,
                       sum(total_cost) AS total_cost,
                       EXTRACT(EPOCH FROM (max(end_time) - min(start_time))) * 1000
                         AS latency_ms
                  FROM observations
                 WHERE project_id = t.project_id AND trace_id = t.id
               ) o ON TRUE
          LEFT JOIN LATERAL (
                SELECT jsonb_object_agg(name, avg_value) AS scores
                  FROM (SELECT name, avg(value) AS avg_value
                          FROM scores
                         WHERE project_id = t.project_id AND trace_id = t.id
                         GROUP BY name) x
               ) s ON TRUE
         WHERE {' AND '.join(conditions)}
         ORDER BY t.timestamp DESC
         LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """, *params)

    return {"data": [
        {**dict(r), "scores": _json(r["scores"]),
         "total_cost": float(r["total_cost"]) if r["total_cost"] is not None else None}
        for r in rows
    ]}


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str, request: Request,
                    ctx: AuthContext = Depends(auth)):
    """One trace, with its observations as a TREE and its scores.

    Returned nested rather than flat so a UI can render the waterfall without
    reconstructing parent/child relationships client-side.
    """
    pool = request.app.state.pool
    trace = await pool.fetchrow(
        "SELECT * FROM traces WHERE project_id = $1 AND id = $2",
        ctx.project_id, trace_id,
    )
    # 404 rather than 403 for a trace in another project: telling an attacker
    # that an id exists but is not theirs is an information leak.
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")

    # All spans, flat, assembled into a tree below. A recursive CTE walking down
    # from the roots silently dropped any span whose parent has not arrived
    # (out-of-order delivery, or a dead-lettered parent) along with its subtree.
    # Those spans are returned at the top level with parent_missing=true.
    rows = await pool.fetch("""
        SELECT * FROM observations
         WHERE project_id = $1 AND trace_id = $2
         ORDER BY start_time, id
    """, ctx.project_id, trace_id)

    nodes = {}
    for row in rows:
        nodes[row["id"]] = {
            "id": row["id"], "parent_id": row["parent_observation_id"],
            "type": row["type"], "name": row["name"],
            "start_time": row["start_time"], "end_time": row["end_time"],
            "latency_ms": row["latency_ms"],
            "time_to_first_token_ms": row["time_to_first_token_ms"],
            "level": row["level"], "status_message": row["status_message"],
            "input": row["input"], "output": row["output"],
            "model": row["provided_model_name"],
            "usage_details": _json(row["usage_details"]),
            "cost_details": _json(row["cost_details"]),
            "total_cost": float(row["total_cost"]) if row["total_cost"] is not None
                          else None,
            "metadata": _json(row["metadata"]),
            "parent_missing": False,
            "children": [],
        }

    roots = []
    for node in nodes.values():
        parent = nodes.get(node["parent_id"])
        if parent is None:
            node["parent_missing"] = node["parent_id"] is not None
            roots.append(node)
        else:
            parent["children"].append(node)

    scores = await pool.fetch("""
        SELECT s.id, s.name, s.value, s.string_value, s.data_type, s.source,
               s.comment, s.observation_id, s.timestamp,
               e.version AS evaluator_version
          FROM scores s
          LEFT JOIN evaluator_versions e ON e.id = s.evaluator_version_id
         WHERE s.project_id = $1 AND s.trace_id = $2
         ORDER BY s.timestamp
    """, ctx.project_id, trace_id)

    total_cost = sum(n["total_cost"] or 0 for n in nodes.values())
    return {
        "id": trace["id"], "name": trace["name"], "user_id": trace["user_id"],
        "session_id": trace["session_id"], "timestamp": trace["timestamp"],
        "release": trace["release"], "version": trace["version"],
        "tags": trace["tags"], "is_experiment": trace["is_experiment"],
        "input": trace["input"], "output": trace["output"],
        "metadata": _json(trace["metadata"]),
        "span_count": len(nodes),
        "total_cost": total_cost or None,
        "observations": roots,
        "scores": [dict(s) for s in scores],
    }


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class DatasetIn(BaseModel):
    name: str
    description: str | None = None
    metadata: dict | None = None


class DatasetItemIn(BaseModel):
    input: Any
    expected_output: Any = None
    metadata: dict | None = None
    source_trace_id: str | None = None
    id: str | None = None


class RunIn(BaseModel):
    name: str
    description: str | None = None
    metadata: dict | None = None


class RunItemIn(BaseModel):
    dataset_item_id: str
    trace_id: str
    observation_id: str | None = None


async def _dataset_id(pool, project_id: str, name: str) -> str:
    row = await pool.fetchrow(
        "SELECT id FROM datasets WHERE project_id = $1 AND name = $2",
        project_id, name,
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"no dataset named {name!r}")
    return row["id"]


@app.post("/datasets", status_code=201)
async def create_dataset(body: DatasetIn, request: Request,
                         ctx: AuthContext = Depends(write_auth)):
    row = await request.app.state.pool.fetchrow("""
        INSERT INTO datasets (project_id, id, name, description, metadata,
                              created_at, updated_at)
        VALUES ($1, gen_random_uuid()::text, $2, $3, $4::jsonb, now(), now())
        ON CONFLICT (project_id, name) DO UPDATE
              SET description = COALESCE(EXCLUDED.description, datasets.description),
                  updated_at = now()
        RETURNING id, name, description
    """, ctx.project_id, body.name, body.description,
         json.dumps(body.metadata) if body.metadata else None)
    return dict(row)


@app.get("/datasets")
async def list_datasets(request: Request, ctx: AuthContext = Depends(auth)):
    rows = await request.app.state.pool.fetch("""
        SELECT d.id, d.name, d.description, d.created_at,
               (SELECT count(*) FROM dataset_items i
                 WHERE i.project_id = d.project_id AND i.dataset_id = d.id
                   AND i.status = 'ACTIVE') AS item_count,
               (SELECT count(*) FROM dataset_runs r
                 WHERE r.project_id = d.project_id AND r.dataset_id = d.id) AS run_count
          FROM datasets d WHERE d.project_id = $1 ORDER BY d.created_at DESC
    """, ctx.project_id)
    return {"data": [dict(r) for r in rows]}


@app.post("/datasets/{name}/items", status_code=201)
async def add_item(name: str, body: DatasetItemIn, request: Request,
                   ctx: AuthContext = Depends(write_auth)):
    pool = request.app.state.pool
    dataset_id = await _dataset_id(pool, ctx.project_id, name)
    row = await pool.fetchrow("""
        INSERT INTO dataset_items
            (project_id, id, dataset_id, input, expected_output, metadata,
             source_trace_id, status, created_at, updated_at)
        VALUES ($1, COALESCE($2, gen_random_uuid()::text), $3, $4::jsonb, $5::jsonb,
                $6::jsonb, $7, 'ACTIVE', now(), now())
        ON CONFLICT (project_id, id) DO UPDATE
              SET input = EXCLUDED.input,
                  expected_output = EXCLUDED.expected_output,
                  updated_at = now()
        RETURNING id
    """, ctx.project_id, body.id, dataset_id,
         json.dumps(body.input), json.dumps(body.expected_output),
         json.dumps(body.metadata) if body.metadata else None, body.source_trace_id)
    return {"id": row["id"]}


@app.post("/datasets/{name}/items/from-trace/{trace_id}", status_code=201)
async def add_item_from_trace(name: str, trace_id: str, request: Request,
                              ctx: AuthContext = Depends(write_auth)):
    """Turn a production trace into a regression test.

    This is the endpoint that makes datasets actually get built. Datasets that
    have to be authored from scratch stay empty; datasets grown from traces you
    already saw go wrong do not.
    """
    pool = request.app.state.pool
    dataset_id = await _dataset_id(pool, ctx.project_id, name)
    trace = await pool.fetchrow("""
        SELECT t.input,
               COALESCE(t.output, (SELECT o.output FROM observations o
                                    WHERE o.project_id = t.project_id
                                      AND o.trace_id = t.id
                                      AND o.parent_observation_id IS NULL
                                    ORDER BY o.start_time LIMIT 1)) AS output
          FROM traces t WHERE t.project_id = $1 AND t.id = $2
    """, ctx.project_id, trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")

    row = await pool.fetchrow("""
        INSERT INTO dataset_items
            (project_id, id, dataset_id, input, expected_output, source_trace_id,
             status, created_at, updated_at)
        VALUES ($1, gen_random_uuid()::text, $2, $3::jsonb, $4::jsonb, $5,
                'ACTIVE', now(), now())
        RETURNING id
    """, ctx.project_id, dataset_id, json.dumps(trace["input"]),
         json.dumps(trace["output"]), trace_id)
    return {"id": row["id"], "source_trace_id": trace_id}


@app.get("/datasets/{name}/items")
async def list_items(name: str, request: Request, ctx: AuthContext = Depends(auth),
                     limit: int = Query(500, ge=1, le=5000)):
    pool = request.app.state.pool
    dataset_id = await _dataset_id(pool, ctx.project_id, name)
    rows = await pool.fetch("""
        SELECT id, input, expected_output, metadata, source_trace_id
          FROM dataset_items
         WHERE project_id = $1 AND dataset_id = $2 AND status = 'ACTIVE'
         ORDER BY created_at LIMIT $3
    """, ctx.project_id, dataset_id, limit)
    return {"data": [
        {"id": r["id"], "input": _json(r["input"]),
         "expected_output": _json(r["expected_output"]),
         "metadata": _json(r["metadata"]), "source_trace_id": r["source_trace_id"]}
        for r in rows
    ]}


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
@app.post("/datasets/{name}/runs", status_code=201)
async def create_run(name: str, body: RunIn, request: Request,
                     ctx: AuthContext = Depends(write_auth)):
    pool = request.app.state.pool
    dataset_id = await _dataset_id(pool, ctx.project_id, name)
    row = await pool.fetchrow("""
        INSERT INTO dataset_runs (project_id, id, dataset_id, name, description,
                                  metadata, created_at)
        VALUES ($1, gen_random_uuid()::text, $2, $3, $4, $5::jsonb, now())
        ON CONFLICT (project_id, dataset_id, name) DO UPDATE SET description =
              COALESCE(EXCLUDED.description, dataset_runs.description)
        RETURNING id, name
    """, ctx.project_id, dataset_id, body.name, body.description,
         json.dumps(body.metadata) if body.metadata else None)
    return dict(row)


@app.post("/datasets/{name}/runs/{run_name}/items", status_code=201)
async def link_run_item(name: str, run_name: str, body: RunItemIn, request: Request,
                        ctx: AuthContext = Depends(write_auth)):
    """Record that running dataset item X during run Y produced trace Z."""
    pool = request.app.state.pool
    dataset_id = await _dataset_id(pool, ctx.project_id, name)
    run = await pool.fetchrow("""
        SELECT id FROM dataset_runs
         WHERE project_id = $1 AND dataset_id = $2 AND name = $3
    """, ctx.project_id, dataset_id, run_name)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run named {run_name!r}")

    row = await pool.fetchrow("""
        INSERT INTO dataset_run_items
            (project_id, id, dataset_run_id, dataset_item_id, trace_id,
             observation_id, created_at)
        VALUES ($1, gen_random_uuid()::text, $2, $3, $4, $5, now())
        ON CONFLICT (project_id, dataset_run_id, dataset_item_id)
              DO UPDATE SET trace_id = EXCLUDED.trace_id
        RETURNING id
    """, ctx.project_id, run["id"], body.dataset_item_id, body.trace_id,
         body.observation_id)
    return {"id": row["id"]}


@app.get("/datasets/{name}/runs")
async def list_runs(name: str, request: Request, ctx: AuthContext = Depends(auth)):
    pool = request.app.state.pool
    dataset_id = await _dataset_id(pool, ctx.project_id, name)
    rows = await pool.fetch("""
        SELECT r.name, r.description, r.metadata, r.created_at,
               count(ri.id) AS item_count
          FROM dataset_runs r
          LEFT JOIN dataset_run_items ri
                 ON ri.project_id = r.project_id AND ri.dataset_run_id = r.id
         WHERE r.project_id = $1 AND r.dataset_id = $2
         GROUP BY r.id, r.name, r.description, r.metadata, r.created_at
         ORDER BY r.created_at DESC
    """, ctx.project_id, dataset_id)
    return {"data": [{**dict(r), "metadata": _json(r["metadata"])} for r in rows]}


# --------------------------------------------------------------------------- #
# Comparison — the headline endpoint
# --------------------------------------------------------------------------- #
@app.get("/datasets/{name}/compare")
async def compare_runs(name: str, request: Request,
                       ctx: AuthContext = Depends(auth),
                       runs: str = Query(..., description="comma-separated run names")):
    """Compare runs of one dataset: scores, cost and latency, plus per-item diffs.

    Scores reach a run through dataset_run_items.trace_id (or dataset_run_item_id
    for scores anchored to the run item itself), so the SAME judge and the SAME
    score rows serve live traffic and offline experiments — there is no separate
    experiment scoring path to keep in sync.
    """
    pool = request.app.state.pool
    dataset_id = await _dataset_id(pool, ctx.project_id, name)
    run_names = [r.strip() for r in runs.split(",") if r.strip()]
    if len(run_names) < 2:
        raise HTTPException(status_code=400, detail="pass at least two run names")

    # Totals and scores are aggregated in separate queries. Joining scores into
    # the totals split each run into one group per score name, so items without
    # a score yet (still settling, or a judge ERROR) formed their own group and
    # overwrote the run's item count and average cost.
    totals = await pool.fetch("""
        SELECT r.name AS run,
               count(*)            AS items,
               avg(agg.total_cost) AS avg_cost,
               sum(agg.total_cost) AS total_cost,
               avg(agg.latency_ms) AS avg_latency_ms
          FROM dataset_runs r
          JOIN dataset_run_items ri
            ON ri.project_id = r.project_id AND ri.dataset_run_id = r.id
          LEFT JOIN LATERAL (
                SELECT sum(o.total_cost) AS total_cost,
                       EXTRACT(EPOCH FROM (max(o.end_time) - min(o.start_time))) * 1000
                         AS latency_ms
                  FROM observations o
                 WHERE o.project_id = ri.project_id AND o.trace_id = ri.trace_id
               ) agg ON TRUE
         WHERE r.project_id = $1 AND r.dataset_id = $2 AND r.name = ANY($3::text[])
         GROUP BY r.name
    """, ctx.project_id, dataset_id, run_names)

    score_rows = await pool.fetch("""
        SELECT r.name AS run, s.name AS score_name,
               avg(s.value) AS avg_score, count(s.id) AS score_count
          FROM dataset_runs r
          JOIN dataset_run_items ri
            ON ri.project_id = r.project_id AND ri.dataset_run_id = r.id
          JOIN scores s
            ON s.project_id = ri.project_id
           AND (s.trace_id = ri.trace_id OR s.dataset_run_item_id = ri.id)
         WHERE r.project_id = $1 AND r.dataset_id = $2 AND r.name = ANY($3::text[])
         GROUP BY r.name, s.name
    """, ctx.project_id, dataset_id, run_names)

    runs_out: dict[str, dict] = {n: {"run": n, "items": 0, "scores": {},
                                     "avg_cost": None, "total_cost": None,
                                     "avg_latency_ms": None} for n in run_names}
    for row in totals:
        entry = runs_out[row["run"]]
        entry["items"] = row["items"]
        if row["avg_cost"] is not None:
            entry["avg_cost"] = float(row["avg_cost"])
            entry["total_cost"] = float(row["total_cost"])
        if row["avg_latency_ms"] is not None:
            entry["avg_latency_ms"] = float(row["avg_latency_ms"])
    for row in score_rows:
        runs_out[row["run"]]["scores"][row["score_name"]] = {
            "mean": float(row["avg_score"]) if row["avg_score"] is not None else None,
            "n": row["score_count"],
        }

    # Per-item, so a mean that barely moved can still be shown to hide a big
    # regression on a handful of cases — which is usually what actually matters.
    items = await pool.fetch("""
        SELECT di.id AS item_id, di.input, di.expected_output,
               r.name AS run, ri.trace_id, s.name AS score_name, s.value
          FROM dataset_runs r
          JOIN dataset_run_items ri
            ON ri.project_id = r.project_id AND ri.dataset_run_id = r.id
          JOIN dataset_items di
            ON di.project_id = ri.project_id AND di.id = ri.dataset_item_id
          LEFT JOIN scores s
            ON s.project_id = ri.project_id
           AND (s.trace_id = ri.trace_id OR s.dataset_run_item_id = ri.id)
         WHERE r.project_id = $1 AND r.dataset_id = $2 AND r.name = ANY($3::text[])
         ORDER BY di.created_at
    """, ctx.project_id, dataset_id, run_names)

    per_item: dict[str, dict] = {}
    for row in items:
        entry = per_item.setdefault(row["item_id"], {
            "item_id": row["item_id"], "input": _json(row["input"]), "runs": {},
        })
        run_entry = entry["runs"].setdefault(row["run"], {"trace_id": row["trace_id"],
                                                          "scores": {}})
        if row["score_name"]:
            run_entry["scores"][row["score_name"]] = float(row["value"]) \
                if row["value"] is not None else None

    baseline, *rest = run_names
    deltas = {}
    for other in rest:
        diff = {}
        for score_name in runs_out[baseline]["scores"]:
            a = runs_out[baseline]["scores"].get(score_name, {}).get("mean")
            b = runs_out[other]["scores"].get(score_name, {}).get("mean")
            if a is not None and b is not None:
                diff[score_name] = round(b - a, 4)
        cost_a, cost_b = runs_out[baseline]["avg_cost"], runs_out[other]["avg_cost"]
        if cost_a and cost_b:
            diff["avg_cost_pct"] = round((cost_b - cost_a) / cost_a * 100, 1)
        deltas[f"{other} vs {baseline}"] = diff

    return {"dataset": name, "runs": list(runs_out.values()),
            "deltas": deltas, "items": list(per_item.values())}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@app.get("/metrics/daily")
async def daily_metrics(request: Request, ctx: AuthContext = Depends(auth),
                        days: int = Query(14, ge=1, le=90)):
    rows = await request.app.state.pool.fetch("""
        SELECT date_trunc('day', t.timestamp) AS day,
               count(DISTINCT t.id) AS traces,
               sum(o.total_cost) AS cost,
               avg(o.latency_ms) AS avg_latency_ms,
               avg(sc.value) AS avg_score
          FROM traces t
          LEFT JOIN observations o
                 ON o.project_id = t.project_id AND o.trace_id = t.id
          LEFT JOIN scores sc
                 ON sc.project_id = t.project_id AND sc.trace_id = t.id
         WHERE t.project_id = $1
           AND t.timestamp >= now() - ($2 || ' days')::interval
           AND NOT t.is_experiment
         GROUP BY 1 ORDER BY 1
    """, ctx.project_id, str(days))
    return {"data": [
        {"day": r["day"], "traces": r["traces"],
         "cost": float(r["cost"]) if r["cost"] is not None else None,
         "avg_latency_ms": float(r["avg_latency_ms"])
                           if r["avg_latency_ms"] is not None else None,
         "avg_score": float(r["avg_score"]) if r["avg_score"] is not None else None}
        for r in rows
    ]}
