"""Benchmark suite — v2.

The v1 suite measured a pipeline that no longer exists: one blocking gRPC call
per trace, a flat trace table, no auth. This one measures what the v2
architecture actually changed, and the first number is the one that matters
most to anyone deciding whether to instrument their app:

  1. SDK overhead per span      how much latency instrumentation adds to YOUR
                                function. v1 blocked the caller on a gRPC round
                                trip per trace; v2 buffers and flushes in the
                                background. This is that difference, measured.
  2. Ingest throughput          events/sec, swept across batch sizes. Batch size
                                1 approximates v1's shape, so the sweep is the
                                argument for batching rather than a claim about it.
  3. End-to-end latency         SDK call -> row readable in Postgres (p50/95/99).
  4. Worker drain rate          how fast a burst backlog clears.
  5. API read latency           trace list, and the trace tree (which the API now
                                assembles in Python rather than in a CTE).
  6. Auth cache effect          request latency with the API-key cache cold vs warm.

    python benchmark/load_test.py                 # human-readable + markdown
    python benchmark/load_test.py --json out.json # machine-readable too
    python benchmark/load_test.py --quick         # smaller N, for a smoke test

Requires the ingestion server, worker and API to be running, and
OPENWEAVE_PUBLIC_KEY / OPENWEAVE_SECRET_KEY to be set.

A NOTE ON REPORTING THESE NUMBERS
---------------------------------
These are single-machine, everything-on-localhost figures. That is a useful
measurement of the code and a meaningless measurement of production capacity,
so the suite prints the machine and configuration it ran on and you should keep
that alongside any number you quote. An unlabelled benchmark number is worth
nothing; a labelled one that you re-ran after a change is worth a lot.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import platform
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "sdk"), os.path.join(ROOT, "benchmark")]

import asyncpg  # noqa: E402
import grpc  # noqa: E402
import httpx  # noqa: E402
import redis.asyncio as redis  # noqa: E402
from google.protobuf.timestamp_pb2 import Timestamp  # noqa: E402
from redis.exceptions import ResponseError  # noqa: E402

import openweave_pb2 as pb  # noqa: E402
import openweave_pb2_grpc as rpc  # noqa: E402
from openweave import OpenWeave  # noqa: E402

GRPC_HOST = os.getenv("OPENWEAVE_HOST", "localhost")
GRPC_PORT = int(os.getenv("OPENWEAVE_PORT", "50051"))
API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
DATABASE_URL = (
    os.getenv("DATABASE_URL", "postgresql://openweave:openweave@localhost:5432/openweave")
    .replace("+asyncpg", "").replace("+psycopg2", "")
)
PUBLIC_KEY = os.getenv("OPENWEAVE_PUBLIC_KEY", "")
SECRET_KEY = os.getenv("OPENWEAVE_SECRET_KEY", "")
STREAM_NAME = os.getenv("STREAM_NAME", "events")
GROUP_NAME = os.getenv("GROUP_NAME", "workers")

AUTH_META = (("authorization", "Basic " + base64.b64encode(
    f"{PUBLIC_KEY}:{SECRET_KEY}".encode()).decode()),)
AUTH_HEADER = {"authorization": AUTH_META[0][1]}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def now_ts() -> Timestamp:
    t = Timestamp()
    t.FromDatetime(datetime.now(timezone.utc))
    return t


def pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile. No interpolation: with N=50 an interpolated p99
    is a fiction, and reporting it as though it were measured is worse than
    reporting the rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * p / 100), len(ordered) - 1)
    return ordered[idx]


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": round(pct(values, 50), 3),
        "p95": round(pct(values, 95), 3),
        "p99": round(pct(values, 99), 3),
        "mean": round(statistics.fmean(values), 3) if values else 0.0,
    }


def make_events(n: int, trace_id: str | None = None,
                kind: str = "GENERATION") -> list:
    """n observation events under one trace — the shape a real agent emits."""
    tid = trace_id or str(uuid.uuid4())
    events = [pb.Event(event_id=str(uuid.uuid4()), timestamp=now_ts(),
                       trace_create=pb.TraceBody(id=tid, name="benchmark",
                                                 timestamp=now_ts()))]
    for _ in range(n):
        body = pb.ObservationBody(
            id=str(uuid.uuid4()), trace_id=tid,
            type=getattr(pb, f"OBSERVATION_TYPE_{kind}"), name="answer",
            start_time=now_ts(), model="gpt-4o-mini",
        )
        body.usage_details["input"] = 480
        body.usage_details["output"] = 120
        events.append(pb.Event(event_id=str(uuid.uuid4()), timestamp=now_ts(),
                               observation_create=body))
    return events


async def stream_backlog(client) -> int:
    """Events the worker has not finished: undelivered plus delivered-but-unacked.

    The PEL alone is not enough — an event no worker has read yet is not pending,
    so an empty PEL can still hide a backlog.
    """
    try:
        info = await client.xinfo_stream(STREAM_NAME)
    except ResponseError:  # stream not created yet
        return 0
    for group in await client.xinfo_groups(STREAM_NAME):
        if group["name"] == GROUP_NAME:
            undelivered = group.get("lag")
            if undelivered is None:  # Redis cannot compute lag after trimming
                undelivered = int(group["last-delivered-id"] != info["last-generated-id"])
            return undelivered + group["pending"]
    return info["length"]


async def wait_for_idle_stream(timeout_s: float = 300.0) -> float:
    """Wait for the worker to catch up, so one stage's backlog is not measured as
    the next stage's latency. Returns the seconds waited."""
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < timeout_s:
            if await stream_backlog(client) == 0:
                return time.perf_counter() - start
            await asyncio.sleep(0.05)
        raise RuntimeError("the worker did not catch up; is it running?")
    finally:
        await client.aclose()


# --------------------------------------------------------------------------- #
# 1. SDK overhead
# --------------------------------------------------------------------------- #
def bench_sdk_overhead(n: int) -> dict:
    """Added latency per decorated call, against a live server.

    Measured as decorated-minus-plain on the same function body, so the payload
    work cancels out and what remains is the SDK's cost to the caller. v1's
    equivalent was a full blocking gRPC round trip per trace; v2 should be an
    in-memory enqueue.
    """
    ow = OpenWeave(public_key=PUBLIC_KEY, secret_key=SECRET_KEY,
                   host=GRPC_HOST, port=GRPC_PORT)

    def payload(x):
        return sum(range(x))

    @ow.observe(type="SPAN", name="bench")
    def instrumented(x):
        return sum(range(x))

    # Warm both paths so JIT-ish effects and the first channel connect are not
    # attributed to the SDK.
    for _ in range(200):
        payload(50); instrumented(50)

    plain, traced = [], []
    with ow.trace(name="overhead-harness"):
        for _ in range(n):
            t0 = time.perf_counter_ns(); payload(50)
            plain.append((time.perf_counter_ns() - t0) / 1000)
            t0 = time.perf_counter_ns(); instrumented(50)
            traced.append((time.perf_counter_ns() - t0) / 1000)

    ow.flush(timeout=30)
    stats = ow.stats
    ow.shutdown()

    overhead = [t - p for t, p in zip(traced, plain)]
    return {
        "plain_us": summarize(plain),
        "instrumented_us": summarize(traced),
        "overhead_us": summarize(overhead),
        "events_sent": stats["sent"],
        "events_dropped": stats["dropped"],
    }


# --------------------------------------------------------------------------- #
# 2. Ingest throughput across batch sizes
# --------------------------------------------------------------------------- #
async def bench_ingest_throughput(total: int, batch_sizes: list[int]) -> dict:
    results = {}
    async with grpc.aio.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}") as channel:
        stub = rpc.IngestionServiceStub(channel)
        for size in batch_sizes:
            events = make_events(total)
            batches = [events[i:i + size] for i in range(0, len(events), size)]
            start = time.perf_counter()
            accepted = 0
            for batch in batches:
                resp = await stub.IngestBatch(
                    pb.IngestBatchRequest(events=batch, sdk="benchmark/2.0"),
                    metadata=AUTH_META, timeout=30)
                accepted += resp.accepted
            elapsed = time.perf_counter() - start
            results[str(size)] = {
                "events": accepted,
                "rpcs": len(batches),
                "seconds": round(elapsed, 3),
                "events_per_sec": round(accepted / elapsed) if elapsed else 0,
            }
    return results


# --------------------------------------------------------------------------- #
# 3. End-to-end latency
# --------------------------------------------------------------------------- #
async def bench_e2e_latency(n: int, timeout_s: float = 15.0) -> dict:
    pool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=1, max_size=4)
    latencies, timeouts = [], 0
    async with grpc.aio.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}") as channel:
        stub = rpc.IngestionServiceStub(channel)
        for _ in range(n):
            obs_id = str(uuid.uuid4())
            tid = str(uuid.uuid4())
            body = pb.ObservationBody(id=obs_id, trace_id=tid,
                                      type=pb.OBSERVATION_TYPE_SPAN,
                                      name="e2e", start_time=now_ts())
            event = pb.Event(event_id=str(uuid.uuid4()), timestamp=now_ts(),
                             observation_create=body)
            t0 = time.perf_counter()
            await stub.IngestBatch(pb.IngestBatchRequest(events=[event]),
                                   metadata=AUTH_META, timeout=30)
            deadline = t0 + timeout_s
            while time.perf_counter() < deadline:
                found = await pool.fetchval(
                    "SELECT 1 FROM observations WHERE id = $1", obs_id)
                if found:
                    latencies.append((time.perf_counter() - t0) * 1000)
                    break
                await asyncio.sleep(0.005)
            else:
                timeouts += 1
    await pool.close()
    return {**summarize(latencies), "timeouts": timeouts, "unit": "ms"}


# --------------------------------------------------------------------------- #
# 4. Worker drain rate
# --------------------------------------------------------------------------- #
async def bench_worker_drain(burst: int, kind: str = "GENERATION",
                             timeout_s: float = 120.0) -> dict:
    """Push a burst as fast as the ingest path allows, then time the drain.

    Reported separately from throughput because they answer different questions:
    throughput is how fast you can accept, drain rate is how fast you can commit.
    A system that accepts faster than it commits just moves the queue.

    Run once for GENERATION and once for plain SPAN events: generations also go
    through cost resolution, so the gap between the two is what pricing costs.
    """
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    events = make_events(burst, kind=kind)

    async with grpc.aio.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}") as channel:
        stub = rpc.IngestionServiceStub(channel)
        start = time.perf_counter()
        for i in range(0, len(events), 200):
            await stub.IngestBatch(pb.IngestBatchRequest(events=events[i:i + 200]),
                                   metadata=AUTH_META, timeout=60)
        accept_seconds = time.perf_counter() - start

        peak = 0
        deadline = time.perf_counter() + timeout_s
        drained_at = None
        while time.perf_counter() < deadline:
            # The worker acks only after its transaction commits, so an empty
            # backlog means every row is readable.
            backlog = await stream_backlog(client)
            peak = max(peak, backlog)
            if backlog == 0:
                drained_at = time.perf_counter()
                break
            await asyncio.sleep(0.02)

    await client.aclose()
    drain_seconds = (drained_at - start) if drained_at else None
    return {
        "kind": kind,
        "events": len(events),
        "accept_seconds": round(accept_seconds, 3),
        "accept_per_sec": round(len(events) / accept_seconds) if accept_seconds else 0,
        "drain_seconds": round(drain_seconds, 3) if drain_seconds else None,
        "drain_per_sec": round(len(events) / drain_seconds) if drain_seconds else None,
        "peak_backlog": peak,
    }


# --------------------------------------------------------------------------- #
# 5. API read latency
# --------------------------------------------------------------------------- #
async def bench_api(n: int) -> dict:
    out: dict = {}
    async with httpx.AsyncClient(base_url=API_BASE, headers=AUTH_HEADER,
                                 timeout=30) as client:
        listing = []
        for _ in range(n):
            t0 = time.perf_counter()
            resp = await client.get("/traces", params={"limit": 50})
            resp.raise_for_status()
            listing.append((time.perf_counter() - t0) * 1000)
        out["list_traces_ms"] = summarize(listing)

        data = (await client.get("/traces", params={"limit": 1})).json()["data"]
        if data:
            trace_id = data[0]["id"]
            detail = []
            for _ in range(n):
                t0 = time.perf_counter()
                resp = await client.get(f"/traces/{trace_id}")
                resp.raise_for_status()
                detail.append((time.perf_counter() - t0) * 1000)
            out["get_trace_tree_ms"] = summarize(detail)
            # Tree latency depends on its size, so report which tree was read.
            out["tree_spans"] = resp.json()["span_count"]
    return out


# --------------------------------------------------------------------------- #
# 6. Auth cache effect
# --------------------------------------------------------------------------- #
async def bench_auth_cache(n: int) -> dict:
    """Cold vs warm key verification.

    The cache exists so the ingest path does not hit Postgres for every request.
    This is the measurement that justifies it — or, if the gap is small on your
    hardware, tells you it does not.
    """
    import hashlib

    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    salt = os.getenv("OPENWEAVE_SALT", "openweave-dev-salt")
    cache_key = "apikey:" + hashlib.sha256((SECRET_KEY + salt).encode()).hexdigest()

    cold, warm = [], []
    async with httpx.AsyncClient(base_url=API_BASE, headers=AUTH_HEADER,
                                 timeout=30) as http:
        for _ in range(max(n // 5, 5)):
            await client.delete(cache_key)
            t0 = time.perf_counter()
            (await http.get("/traces", params={"limit": 1})).raise_for_status()
            cold.append((time.perf_counter() - t0) * 1000)
        for _ in range(n):
            t0 = time.perf_counter()
            (await http.get("/traces", params={"limit": 1})).raise_for_status()
            warm.append((time.perf_counter() - t0) * 1000)
    await client.aclose()

    saved = pct(cold, 50) - pct(warm, 50)
    return {"cold_ms": summarize(cold), "warm_ms": summarize(warm),
            "p50_saved_ms": round(saved, 3)}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def environment() -> dict:
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_count": os.cpu_count(),
        "note": "all services on localhost; not a production capacity figure",
    }


def render(results: dict) -> str:
    env, L = results["environment"], []
    L.append("## Benchmarks\n")
    L.append(f"_{env['platform']}, {env['cpu_count']} cores, Python {env['python']}, "
             f"{env['captured_at']}. All services on localhost — this measures the "
             f"code, not production capacity._\n")

    o = results["sdk_overhead"]
    L.append("**SDK overhead per span** — what instrumentation costs the caller\n")
    L.append("| | p50 | p95 |")
    L.append("|---|---|---|")
    L.append(f"| Plain call | {o['plain_us']['p50']} µs | {o['plain_us']['p95']} µs |")
    L.append(f"| Instrumented | {o['instrumented_us']['p50']} µs | "
             f"{o['instrumented_us']['p95']} µs |")
    L.append(f"| **Added by the SDK** | **{o['overhead_us']['p50']} µs** | "
             f"{o['overhead_us']['p95']} µs |")
    L.append(f"\n{o['events_sent']} events delivered, {o['events_dropped']} dropped.\n")

    L.append("**Ingest throughput by batch size** — batch size 1 is v1's shape\n")
    L.append("| Batch size | RPCs | Events/sec |")
    L.append("|---|---|---|")
    for size, r in results["ingest_throughput"].items():
        L.append(f"| {size} | {r['rpcs']} | {r['events_per_sec']:,} |")

    e = results["e2e_latency"]
    L.append(f"\n**End-to-end latency** (SDK call → readable in Postgres, on a "
             f"drained stream, n={e['n']}): p50 {e['p50']} ms · p95 {e['p95']} ms · "
             f"p99 {e['p99']} ms"
             + (f" · {e['timeouts']} timeouts" if e["timeouts"] else ""))

    drains = results["worker_drain"]
    if drains:
        L.append("\n**Burst drain** — accept rate vs commit rate\n")
        L.append("| Burst | Accepted/sec | Committed in | Committed/sec |")
        L.append("|---|---|---|---|")
        for d in drains.values():
            rate = f"{d['drain_per_sec']:,}" if d["drain_per_sec"] else "timed out"
            L.append(f"| {d['events']:,} {d['kind']} events | {d['accept_per_sec']:,} | "
                     f"{d['drain_seconds']} s | {rate} |")

    a = results["api"]
    if a:
        L.append("\n**API read latency**\n")
        L.append("| Endpoint | p50 | p95 |")
        L.append("|---|---|---|")
        L.append(f"| `GET /traces` (50 rows) | {a['list_traces_ms']['p50']} ms | "
                 f"{a['list_traces_ms']['p95']} ms |")
        if "get_trace_tree_ms" in a:
            spans = a.get("tree_spans")
            label = f"{spans:,}-span tree" if spans else "tree"
            L.append(f"| `GET /traces/{{id}}` ({label}) | "
                     f"{a['get_trace_tree_ms']['p50']} ms | "
                     f"{a['get_trace_tree_ms']['p95']} ms |")

    c = results["auth_cache"]
    L.append(f"\n**API key cache**: p50 {c['cold_ms']['p50']} ms cold vs "
             f"{c['warm_ms']['p50']} ms warm — {c['p50_saved_ms']} ms saved per "
             f"authenticated request.")

    L.append("\n" + interpret(results))
    return "\n".join(L)


def interpret(results: dict) -> str:
    """Say what the numbers mean.

    A table of figures with no reading is a number dump. These are the three
    things someone will ask about, answered in advance — including the one that
    looks like a problem and is not.
    """
    lines = ["**Reading these numbers**\n"]

    tp = results["ingest_throughput"]
    if "1" in tp and "200" in tp and tp["1"]["events_per_sec"]:
        ratio = tp["200"]["events_per_sec"] / tp["1"]["events_per_sec"]
        lines.append(
            f"- Batching is worth **{ratio:.0f}x** here ({tp['1']['events_per_sec']:,} → "
            f"{tp['200']['events_per_sec']:,} events/sec). v1 sent one blocking RPC per "
            f"trace, so the batch-size-1 row is roughly its ceiling.")

    o = results["sdk_overhead"]["overhead_us"]
    lines.append(
        f"- The SDK adds **~{o['p50']:.0f} µs** to a decorated call — building the "
        f"span's create and update events (ids, timestamps, protobuf) and an "
        f"in-memory enqueue; nothing waits on the network. v1 held the caller for a "
        f"full gRPC round trip instead. Argument capture is a small share for small "
        f"arguments but grows with their size, so pass `capture_input=False` on "
        f"functions that take large inputs.")

    e = results["e2e_latency"]
    lines.append(
        f"- End-to-end p50 **{e['p50']:.0f} ms**, p95 {e['p95']:.0f} ms, measured once "
        f"the previous stage's backlog has drained. Without that wait the first "
        f"samples queue behind it, and at n={e['n']} a single slow sample becomes the "
        f"p99. A blocked XREADGROUP returns as soon as an event arrives, so an idle "
        f"worker adds no poll delay.")

    drains = results["worker_drain"]
    gen, span = drains.get("GENERATION", {}), drains.get("SPAN", {})
    if gen.get("drain_per_sec") and span.get("drain_per_sec"):
        lines.append(
            f"- Ingest accepts **{gen['accept_per_sec']:,}/sec**, but the worker commits "
            f"generations at **{gen['drain_per_sec']:,}/sec** and plain spans at "
            f"{span['drain_per_sec']:,}/sec. The gap is cost resolution, which issues "
            f"one UPDATE per priced generation, so that is the first thing to batch if "
            f"commit rate matters. Accepting faster than you commit only moves the "
            f"queue: the commit rate is what the stream's MAXLEN and the worker replica "
            f"count have to be sized against.")

    lines.append(
        "- Throughput and drain vary between runs, and the first run against a fresh "
        "database is noticeably slower. Discard it and report the median of the next "
        "few.")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", metavar="PATH", help="also write raw results here")
    parser.add_argument("--quick", action="store_true", help="smaller N")
    args = parser.parse_args()

    if not PUBLIC_KEY or not SECRET_KEY:
        sys.exit("set OPENWEAVE_PUBLIC_KEY and OPENWEAVE_SECRET_KEY "
                 "(scripts/bootstrap.py prints them)")

    n_overhead = 200 if args.quick else 2000
    n_ingest = 500 if args.quick else 5000
    n_e2e = 15 if args.quick else 50
    n_burst = 500 if args.quick else 5000
    n_api = 20 if args.quick else 100

    print("[1/6] SDK overhead…", flush=True)
    sdk = await asyncio.get_running_loop().run_in_executor(
        None, bench_sdk_overhead, n_overhead)

    print("[2/6] ingest throughput…", flush=True)
    throughput = await bench_ingest_throughput(n_ingest, [1, 25, 200])

    print("[3/6] end-to-end latency…", flush=True)
    await wait_for_idle_stream()
    e2e = await bench_e2e_latency(n_e2e)

    print("[4/6] burst drain…", flush=True)
    drain = {}
    for kind in ("GENERATION", "SPAN"):
        await wait_for_idle_stream()
        drain[kind] = await bench_worker_drain(n_burst, kind)

    print("[5/6] api latency…", flush=True)
    await wait_for_idle_stream()
    api = await bench_api(n_api)

    print("[6/6] auth cache…", flush=True)
    cache = await bench_auth_cache(n_api)

    results = {
        "environment": environment(), "sdk_overhead": sdk,
        "ingest_throughput": throughput, "e2e_latency": e2e,
        "worker_drain": drain, "api": api, "auth_cache": cache,
    }

    print("\n" + render(results) + "\n")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"raw results → {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
