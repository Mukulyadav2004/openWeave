"""Alembic migration environment.

Resolves the database URL from the DATABASE_URL environment variable and targets the
SQLAlchemy metadata defined in ``models.py``.
"""
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

from models import Base

# Alembic Config object, providing access to values within alembic.ini.
config = context.config

# Resolve the URL from the environment so no credentials live in alembic.ini.
# Alembic runs synchronously, so coerce the worker's async driver URL to psycopg2.
database_url = os.getenv("DATABASE_URL")
if database_url:
    database_url = database_url.replace("+asyncpg", "+psycopg2")
    # SQLAlchemy rejects the postgres:// scheme some hosts still hand out.
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]
    config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DBAPI connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
