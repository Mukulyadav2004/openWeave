"""Seed the models + prices tables.

    python scripts/seed_models.py

⚠️  THE NUMBERS BELOW ARE ILLUSTRATIVE DEFAULTS, NOT AUTHORITATIVE PRICES.
    Verify each one against the provider's own pricing page before you quote a
    cost figure anywhere that matters, and re-check when you demo this. Prices
    change often and vary by region and tier. The point of this file is the
    SHAPE — regex matching, per-usage-type rows, start_date — not the values.

Prices are entered per MILLION tokens (how vendors publish them) and divided
down to per-token on insert, which is how the prices table stores them.

Re-runnable: rows are matched on (model_name, project_id) and updated in place.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from models import Model, Price  # noqa: E402

DATABASE_URL = (
    os.getenv("DATABASE_URL", "postgresql://openweave:openweave@localhost:5432/openweave")
    .replace("+asyncpg", "")
)

# Effective date for this price sheet. Later sheets get a newer start_date and
# BOTH rows are kept, so historical traces keep the price that applied then.
EFFECTIVE_FROM = datetime(2026, 1, 1, tzinfo=timezone.utc)

# name, regex, {usage_type: price per 1M tokens}
SEED: list[tuple[str, str, dict[str, str]]] = [
    # --- OpenAI ---------------------------------------------------------- #
    ("gpt-4o",       r"(?i)^gpt-4o(-\d{4}-\d{2}-\d{2})?$",
     {"input": "2.50", "output": "10.00", "cache_read": "1.25"}),
    ("gpt-4o-mini",  r"(?i)^gpt-4o-mini(-\d{4}-\d{2}-\d{2})?$",
     {"input": "0.15", "output": "0.60", "cache_read": "0.075"}),
    ("gpt-4-turbo",  r"(?i)^gpt-4(-turbo)?(-preview)?(-\d{4}-\d{2}-\d{2})?$",
     {"input": "10.00", "output": "30.00"}),
    ("o1",           r"(?i)^o1(-mini|-preview)?(-\d{4}-\d{2}-\d{2})?$",
     {"input": "15.00", "output": "60.00", "reasoning": "60.00"}),

    # --- Anthropic ------------------------------------------------------- #
    ("claude-sonnet", r"(?i)^claude-3[.-]?[57]?-sonnet.*$",
     {"input": "3.00", "output": "15.00",
      "cache_read": "0.30", "cache_write": "3.75"}),
    ("claude-haiku",  r"(?i)^claude-3[.-]?5?-haiku.*$",
     {"input": "0.80", "output": "4.00",
      "cache_read": "0.08", "cache_write": "1.00"}),
    # Claude 3 Opus only. The looser ^claude-3?-?opus also matched newer IDs such
    # as claude-opus-5 and priced them at these rates. Give each newer model its
    # own row and prices instead of widening a pattern.
    ("claude-opus",   r"(?i)^claude-3-opus.*$",
     {"input": "15.00", "output": "75.00",
      "cache_read": "1.50", "cache_write": "18.75"}),

    # --- Google ---------------------------------------------------------- #
    ("gemini-flash",  r"(?i)^gemini-[0-9.]+-flash.*$",
     {"input": "0.075", "output": "0.30"}),
    ("gemini-pro",    r"(?i)^gemini-[0-9.]+-pro.*$",
     {"input": "1.25", "output": "5.00"}),

    # --- Self-hosted ------------------------------------------------------ #
    # Zero-priced on purpose: a local model has no per-token cost, and this row
    # exists so those generations resolve to 0.00 rather than to NULL/"unknown".
    ("ollama-local",  r"(?i)^(llama|mistral|mixtral|qwen|phi|gemma)[0-9a-z.:\-]*$",
     {"input": "0", "output": "0"}),
]

PER_MILLION = Decimal(1_000_000)


def main() -> None:
    engine = create_engine(DATABASE_URL)
    created = updated = 0

    with Session(engine) as session:
        for model_name, pattern, prices in SEED:
            model = session.scalar(
                select(Model).where(
                    Model.model_name == model_name, Model.project_id.is_(None)
                )
            )
            if model is None:
                model = Model(
                    model_name=model_name, project_id=None,
                    match_pattern=pattern, start_date=EFFECTIVE_FROM,
                )
                session.add(model)
                session.flush()
                created += 1
            else:
                model.match_pattern = pattern
                model.start_date = EFFECTIVE_FROM
                updated += 1

            existing = {
                p.usage_type: p
                for p in session.scalars(
                    select(Price).where(Price.model_id == model.id)
                )
            }
            for usage_type, per_million in prices.items():
                per_token = Decimal(per_million) / PER_MILLION
                if usage_type in existing:
                    existing[usage_type].price = per_token
                else:
                    session.add(Price(
                        model_id=model.id, usage_type=usage_type, price=per_token
                    ))
        session.commit()

    print(f"seeded models: {created} created, {updated} updated")
    print("NOTE: prices are illustrative — verify against provider pricing pages.")


if __name__ == "__main__":
    main()
