"""Shared fixtures: local-Docker Postgres, migrated once per session."""

from __future__ import annotations

import os

import psycopg
import pytest

from rag import db as rag_db
from rag import migrate as rag_migrate

LOCAL_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://rag:rag@127.0.0.1:55434/rag_test"
)
MAINT_URL = "postgresql://rag:rag@127.0.0.1:55434/rag"
INIT_REVISION = "20261006_0001"


@pytest.fixture(scope="session")
def test_db_url():
    assert rag_db.is_local_database_url(LOCAL_URL), "refusing non-local test DB"
    with psycopg.connect(MAINT_URL, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = 'rag_test'"
        ).fetchone()
        if not exists:
            conn.execute("CREATE DATABASE rag_test")
    return LOCAL_URL


@pytest.fixture(scope="session")
def migrated_db(test_db_url):
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        conn.execute(
            "DROP TABLE IF EXISTS chunks, messages, documents, conversations,"
            " embedding_cache, answer_cache, tool_cache, prompt_templates,"
            " app_meta, alembic_version, schema_migrations CASCADE"
        )
    applied = rag_migrate.migrate(test_db_url)
    assert applied == [INIT_REVISION]
    return test_db_url
