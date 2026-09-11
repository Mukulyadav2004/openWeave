"""Model matching and cost resolution.

Turns `usage_details` ({"input": 1200, "output": 340, "cache_read": 800}) into
`cost_details` and a `total_cost`, using the `models` and `prices` tables.

Three decisions worth being able to defend:

1.  MATCH BY REGEX, NOT EXACT NAME. Providers ship dated snapshots faster than
    you can seed rows, so one pattern covers a family:
        (?i)^gpt-4o(-\\d{4}-\\d{2}-\\d{2})?$
    Exact-name tables go stale within weeks and silently report zero cost,
    which is worse than reporting none at all.

2.  PRICE AS OF THE OBSERVATION, NOT AS OF NOW. A model row carries a
    start_date; resolution picks the newest row whose start_date is <= the
    observation's start_time. Re-costing a trace from three months ago gives
    the price that applied THEN. Without this, every provider price change
    silently rewrites your historical spend.

3.  DECIMAL, NEVER FLOAT. Token prices are ~1e-8 and get multiplied by counts
    in the millions. Binary floating point accumulates visible error at that
    scale, and "our cost dashboard is off by 3%" is an unpleasant bug to chase.

A project-scoped model always beats a global one, so a team can override a
price without touching the shared table.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import text

logger = logging.getLogger("openweave.pricing")

# Usage keys that are sums of other keys. Pricing them double-counts.
AGGREGATE_USAGE_KEYS = {"total", "total_tokens"}


@dataclass(frozen=True)
class ModelPrice:
    model_id: str
    model_name: str
    project_id: str | None
    pattern: re.Pattern
    start_date: datetime | None
    # usage_type -> price per single token
    prices: dict[str, Decimal] = field(default_factory=dict)


class ModelPriceCache:
    """In-process cache of the model/price tables.

    Refreshed on a TTL rather than per lookup: this sits in the ingestion hot
    path, and the tables change roughly never. A stale entry costs you at most
    `ttl` seconds of old pricing on new rows, which is recoverable by re-running
    resolution — losing throughput to a per-observation query is not.
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        # None, not 0.0: time.monotonic() counts from boot on Linux, so a process
        # started within `ttl` seconds of boot (a fresh CI runner or k8s node)
        # saw monotonic() - 0.0 < ttl, skipped its first load, and priced every
        # generation as unknown until the TTL ran out.
        self._loaded_at: float | None = None
        self._models: list[ModelPrice] = []

    async def refresh(self, conn, force: bool = False) -> None:
        if (not force and self._loaded_at is not None
                and (time.monotonic() - self._loaded_at) < self._ttl):
            return
        rows = (await conn.execute(text("""
            SELECT m.id, m.model_name, m.project_id, m.match_pattern, m.start_date,
                   p.usage_type, p.price
              FROM models m
              LEFT JOIN prices p ON p.model_id = m.id
        """))).mappings().all()

        by_model: dict[str, dict] = {}
        for r in rows:
            entry = by_model.setdefault(r["id"], {
                "model_name": r["model_name"],
                "project_id": r["project_id"],
                "match_pattern": r["match_pattern"],
                "start_date": r["start_date"],
                "prices": {},
            })
            if r["usage_type"] is not None:
                entry["prices"][r["usage_type"]] = Decimal(str(r["price"]))

        compiled: list[ModelPrice] = []
        for model_id, e in by_model.items():
            try:
                pattern = re.compile(e["match_pattern"])
            except re.error as exc:
                # One bad regex must not take down cost resolution for every
                # other model.
                logger.error("model %s has an invalid match_pattern: %s",
                             e["model_name"], exc)
                continue
            compiled.append(ModelPrice(
                model_id=model_id, model_name=e["model_name"],
                project_id=e["project_id"], pattern=pattern,
                start_date=e["start_date"], prices=e["prices"],
            ))

        self._models = compiled
        self._loaded_at = time.monotonic()
        logger.info("loaded %d model price definitions", len(compiled))

    def match(self, model_name: str | None, project_id: str,
              at: datetime | None = None) -> ModelPrice | None:
        """Best price row for this model name, project and point in time."""
        if not model_name:
            return None
        at = at or datetime.now(timezone.utc)
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)

        candidates = [
            m for m in self._models
            if m.project_id in (None, project_id)
            and m.pattern.match(model_name)
            and (m.start_date is None or m.start_date <= at)
        ]
        if not candidates:
            return None
        # Project override wins; then the most recent applicable price.
        return max(
            candidates,
            key=lambda m: (
                m.project_id is not None,
                m.start_date or datetime.min.replace(tzinfo=timezone.utc),
            ),
        )


def compute_cost(usage: dict | None, price: ModelPrice | None
                 ) -> tuple[dict | None, Decimal | None]:
    """(cost_details, total_cost) for one observation.

    Returns (None, None) when the model is unknown or nothing is priceable —
    deliberately, so an unpriced observation reads as "we don't know" rather
    than as a confident zero.
    """
    if not usage or price is None or not price.prices:
        return None, None

    details: dict[str, float] = {}
    total = Decimal(0)
    for usage_type, count in usage.items():
        if usage_type in AGGREGATE_USAGE_KEYS:
            continue
        unit = price.prices.get(usage_type)
        if unit is None:
            continue
        try:
            amount = Decimal(int(count)) * unit
        except (TypeError, ValueError):
            continue
        details[usage_type] = float(amount)
        total += amount

    if not details:
        return None, None
    return details, total


# --------------------------------------------------------------------------- #
# Batch resolution
# --------------------------------------------------------------------------- #
# Two parallel text[] arrays joined via unnest, NOT `(project_id, id) = ANY(:keys)`.
# asyncpg cannot bind a list of tuples: PostgreSQL has no input syntax for an
# anonymous composite type, so the row-constructor form fails at the driver.
# unnest of two arrays is the portable way to pass a set of composite keys.
RESOLVE_SELECT = text("""
    SELECT o.project_id, o.id, o.provided_model_name, o.start_time, o.usage_details
      FROM observations o
      JOIN unnest(CAST(:pids AS text[]), CAST(:ids AS text[]))
             AS k(project_id, id)
        ON o.project_id = k.project_id AND o.id = k.id
     WHERE o.type = 'GENERATION'
       AND o.usage_details IS NOT NULL
       AND o.provided_cost_details IS NULL
""")

RESOLVE_UPDATE = text("""
    UPDATE observations
       SET internal_model_id = :model_id,
           cost_details      = CAST(:cost_details AS jsonb),
           total_cost        = :total_cost
     WHERE project_id = :project_id AND id = :id
""")


async def resolve_costs(conn, cache: ModelPriceCache, keys: set[tuple[str, str]]) -> int:
    """Price every GENERATION among `keys` that the client did not price itself.

    Runs AFTER the batch is written rather than during row building, because
    model name and token usage often arrive in different events — the model on
    the create, the usage on the update. Resolving at write time would price
    half the generations at zero.
    """
    if not keys:
        return 0
    await cache.refresh(conn)

    import json

    ordered = list(keys)
    rows = (await conn.execute(RESOLVE_SELECT, {
        "pids": [k[0] for k in ordered],
        "ids": [k[1] for k in ordered],
    })).mappings().all()

    updated = 0
    for row in rows:
        price = cache.match(
            row["provided_model_name"], row["project_id"], row["start_time"]
        )
        details, total = compute_cost(row["usage_details"], price)
        if details is None:
            if row["provided_model_name"]:
                logger.debug("no price for model %r", row["provided_model_name"])
            continue
        await conn.execute(RESOLVE_UPDATE, {
            "model_id": price.model_id,
            "cost_details": json.dumps(details),
            "total_cost": total,
            "project_id": row["project_id"],
            "id": row["id"],
        })
        updated += 1
    return updated
