"""Alembic environment: URL from DATABASE_URL env, metadata from rag.models."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Ensure `rag` is importable when alembic runs from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag import db as rag_db
from rag.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    configured = config.get_main_option("sqlalchemy.url")
    if config.config_file_name is not None:
        # CLI (`alembic upgrade head`): env var overrides the ini placeholder.
        url = os.environ.get("DATABASE_URL") or configured
    else:
        # Programmatic (`rag.migrate.migrate`): explicit argument wins.
        url = configured
    return rag_db.sqlalchemy_url(url)


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    config.set_main_option("sqlalchemy.url", get_url())
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
