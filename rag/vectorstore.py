"""LangChain ``VectorStore`` over the app's own ``chunks`` table.

One store, not two: chunk text, 2048-dim embeddings, and the full-text
``tsv`` column live in ``chunks``. LangChain's ``PGVector`` tables would
duplicate every embedding while adding nothing (its HNSW management cannot
run above pgvector's 2000-dim cap anyway). Hybrid search lands in T-004.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def to_vector_literal(vector: list[float]) -> str:
    """Render floats as a pgvector text literal."""
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


class ChunksVectorStore(VectorStore):
    """Exact-cosine store on ``chunks``; score is cosine similarity."""

    def __init__(self, database_url: str, embeddings: Embeddings) -> None:
        self.database_url = database_url
        self._embeddings = embeddings

    @property
    def embeddings(self) -> Embeddings:
        """The injected embeddings object."""
        return self._embeddings

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> ChunksVectorStore:
        """Build a store from texts (DATABASE_URL fallback)."""
        database_url = kwargs.get("database_url") or os.environ.get(
            "DATABASE_URL"
        )
        if not database_url:
            raise ValueError(
                "ChunksVectorStore needs database_url "
                "(kwarg or DATABASE_URL env)."
            )
        store = cls(database_url, embedding)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and insert chunk rows; return new ids."""
        if ids is not None:
            raise ValueError("chunks use database-generated ids; omit ids")
        items = list(texts)
        if not items:
            return []
        metas = list(metadatas) if metadatas is not None else [{}] * len(items)
        if len(metas) != len(items):
            raise ValueError("texts and metadatas must have equal length")
        for meta in metas:
            if "document_id" not in meta or "ord" not in meta:
                raise ValueError(
                    "each metadata needs 'document_id' and 'ord'"
                )
        vectors = self._embeddings.embed_documents(items)
        new_ids: list[str] = []
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            for text, meta, vector in zip(items, metas, vectors):
                row = conn.execute(
                    "INSERT INTO chunks"
                    " (document_id, ord, text, token_count, embedding,"
                    " metadata)"
                    " VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)"
                    " RETURNING id",
                    (
                        meta["document_id"],
                        meta["ord"],
                        text,
                        meta.get("token_count"),
                        to_vector_literal(vector),
                        psycopg.types.json.Json(meta.get("extra", {})),
                    ),
                ).fetchone()
                assert row is not None
                new_ids.append(str(row[0]))
        return new_ids

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Top-k chunks with cosine similarity."""
        vector = self._embeddings.embed_query(query)
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                "SELECT c.id, c.text, c.ord, c.document_id, d.filename,"
                " 1 - (c.embedding <=> %s::vector) AS similarity"
                " FROM chunks c JOIN documents d ON d.id = c.document_id"
                " WHERE c.embedding IS NOT NULL"
                " ORDER BY c.embedding <=> %s::vector LIMIT %s",
                (to_vector_literal(vector), to_vector_literal(vector), k),
            ).fetchall()
        return [
            (
                Document(
                    page_content=row[1],
                    metadata={
                        "chunk_id": row[0],
                        "ord": row[2],
                        "document_id": row[3],
                        "filename": row[4],
                    },
                ),
                float(row[5]),
            )
            for row in rows
        ]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Top-k chunk Documents."""
        return [
            doc
            for doc, _ in self.similarity_search_with_score(query, k=k)
        ]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """Delete chunks by id."""
        if not ids:
            return True
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM chunks WHERE id = ANY(%s::bigint[])",
                ([int(i) for i in ids],),
            )
        return True
