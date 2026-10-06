"""T-001: Alembic migrations + pools (local Docker only).

These tests only ever run against a localhost Postgres. The URL guard test
below fails closed: any non-localhost URL is rejected, so the suite can never
touch the Neon production database. DB fixtures live in conftest.py.
"""

from __future__ import annotations

import psycopg
from sqlalchemy import text

from rag import db as rag_db
from rag import migrate as rag_migrate
from rag.models import Chunk, Document

APP_TABLES = {
    "documents",
    "chunks",
    "conversations",
    "messages",
    "embedding_cache",
    "answer_cache",
    "tool_cache",
    "prompt_templates",
    "app_meta",
}

DIM = 2048


def test_url_guard_rejects_non_localhost(test_db_url):
    assert rag_db.is_local_database_url(test_db_url)
    assert rag_db.is_local_database_url("postgresql://u:p@localhost:5432/x")
    assert not rag_db.is_local_database_url(
        "postgresql://owner:secret@ep-xyz-pooler.c-6.aws.neon.tech/neondb"
        "?sslmode=require"
    )
    assert not rag_db.is_local_database_url("postgresql://u:p@10.0.0.5:5432/x")


def test_sqlalchemy_url_uses_psycopg_dialect():
    assert (
        rag_db.sqlalchemy_url("postgresql://u:p@h:5432/x")
        == "postgresql+psycopg://u:p@h:5432/x"
    )


def test_migrate_is_idempotent(migrated_db):
    assert rag_migrate.migrate(migrated_db) == []


def test_pool_connects_and_health(test_db_url):
    with rag_db.get_pool(test_db_url) as pool:
        assert rag_db.health(pool) is True


def test_engine_connects_and_health(test_db_url):
    with (
        rag_db.get_engine(test_db_url) as engine,
        rag_db.get_session(engine) as session,
    ):
        assert rag_db.health_engine(engine) is True
        assert session.execute(text("SELECT 1")).scalar() == 1


def test_all_app_tables_exist(migrated_db):
    with psycopg.connect(migrated_db) as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert APP_TABLES <= names


def test_corpus_version_seeded(migrated_db):
    # Shared DB: ingestion tests bump the version, so this pins "seeded
    # and sane" (>= 1); the exact seed value 1 is covered by fresh-migrate
    # runs (T-001 log) and relative-bump tests elsewhere.
    with psycopg.connect(migrated_db) as conn:
        value = conn.execute(
            "SELECT value FROM app_meta WHERE key = 'corpus_version'"
        ).fetchone()
    assert value is not None and int(value[0]) >= 1


def test_orm_roundtrip(migrated_db):
    with (
        rag_db.get_engine(migrated_db) as engine,
        rag_db.get_session(engine) as session,
    ):
        doc = Document(filename="orm.txt", sha256="orm-sha")
        session.add(doc)
        session.flush()
        session.add(Chunk(document_id=doc.id, ord=0, text="hello"))
        session.commit()
        got = session.get(Document, doc.id)
        assert got is not None and got.filename == "orm.txt"
        assert got.status == "ready"
        session.delete(got)
        session.commit()
        assert session.get(Document, doc.id) is None


def test_cosine_similarity_top1(migrated_db):
    e1 = "[" + ",".join(["1.0"] + ["0.0"] * (DIM - 1)) + "]"
    e2 = "[" + ",".join(["0.0"] * (DIM - 1) + ["1.0"]) + "]"
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        doc = conn.execute(
            "INSERT INTO documents (filename, sha256) VALUES ('t.txt', 'abc')"
            " RETURNING id"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO chunks (document_id, ord, text, embedding)"
            " VALUES (%s, 0, 'first', %s::vector),"
            " (%s, 1, 'second', %s::vector)",
            (doc, e1, doc, e2),
        )
        top = conn.execute(
            "SELECT text FROM chunks WHERE document_id = %s"
            " ORDER BY embedding <=> %s::vector LIMIT 1",
            (doc, e1),
        ).fetchone()[0]
        conn.execute("DELETE FROM documents WHERE id = %s", (doc,))
    assert top == "first"
