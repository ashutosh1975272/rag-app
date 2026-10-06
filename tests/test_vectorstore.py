"""VectorStore tests on local Docker DB with deterministic stub embeddings.

No network, no keys: a one-hot stub over 2048 dims proves ranking, and the
store round-trips through the real ``chunks`` table.
"""

from __future__ import annotations

import psycopg
import pytest
from langchain_core.embeddings import Embeddings

from rag.models import EMBEDDING_DIM
from rag.vectorstore import ChunksVectorStore


class StubEmbeddings(Embeddings):
    """One-hot by marker word: 'alpha' -> e0, 'beta' -> e2047."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * EMBEDDING_DIM
        low = text.lower()
        if "alpha" in low:
            v[0] = 1.0
        elif "beta" in low:
            v[-1] = 1.0
        else:
            v[1] = 1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def doc_id(migrated_db):
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        row = conn.execute(
            "INSERT INTO documents (filename, sha256)"
            " VALUES ('vec.txt', 'vec-sha')"
            " ON CONFLICT (sha256) DO UPDATE SET filename = 'vec.txt'"
            " RETURNING id"
        ).fetchone()
        assert row is not None
        yield row[0]
        conn.execute("DELETE FROM documents WHERE id = %s", (row[0],))


@pytest.fixture
def store(migrated_db):
    return ChunksVectorStore(migrated_db, StubEmbeddings())


def test_add_and_top1_ranking(store, doc_id):
    ids = store.add_texts(
        ["alpha first chunk", "beta second chunk"],
        metadatas=[
            {"document_id": doc_id, "ord": 0},
            {"document_id": doc_id, "ord": 1},
        ],
    )
    assert len(ids) == 2
    try:
        hits = store.similarity_search_with_score("alpha query", k=2)
        assert len(hits) == 2
        top, top_score = hits[0]
        assert top.page_content == "alpha first chunk"
        assert top.metadata["ord"] == 0
        assert top.metadata["document_id"] == doc_id
        assert top.metadata["filename"] == "vec.txt"
        assert top_score == pytest.approx(1.0)
        assert hits[0][1] > hits[1][1]  # similarity desc
        docs = store.similarity_search("beta query", k=1)
        assert docs[0].page_content == "beta second chunk"
    finally:
        assert store.delete(ids) is True
        remaining = store.similarity_search("alpha query", k=100)
        assert all(
            d.page_content not in {"alpha first chunk", "beta second chunk"}
            for d in remaining
        )


def test_add_requires_document_metadata(store):
    with pytest.raises(ValueError, match="document_id"):
        store.add_texts(["x"], metadatas=[{"ord": 0}])
    with pytest.raises(ValueError, match="equal length"):
        store.add_texts(["x", "y"], metadatas=[{"document_id": 1, "ord": 0}])


def test_delete_empty_is_noop(store):
    assert store.delete([]) is True
    assert store.delete(None) is True


def test_from_texts_needs_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="database_url"):
        ChunksVectorStore.from_texts(
            ["x"],
            StubEmbeddings(),
            metadatas=[{"document_id": 1, "ord": 0}],
        )
