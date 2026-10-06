"""Retrieval: vector cosine search plus hybrid (vector + full-text, RRF).

No reranker: the 80-model NIM list contains no rerank function for this key
(checked live 2026-10-06), so hybrid RRF is the top tier. Revisit if one
appears.
"""

from __future__ import annotations

import psycopg

from rag.vectorstore import to_vector_literal

RRF_K = 60


def search(
    question: str,
    *,
    database_url: str,
    client,
    top_k: int = 6,
    mode: str = "vector",
) -> list[dict]:
    """Rank chunks for *question*.

    Returns ``[{content, doc_name, chunk_index, score, mode}]`` ordered
    best-first. ``mode`` is ``vector`` (cosine similarity) or ``hybrid``
    (RRF of vector rank + full-text rank). Blank questions and empty
    corpora return ``[]``.
    """
    if mode not in ("vector", "hybrid"):
        raise ValueError(f"mode must be 'vector' or 'hybrid', got {mode!r}")
    if not question or not question.strip():
        return []
    vector = client.embed([question], "query")[0]
    if mode == "vector":
        return _search_vector(database_url, vector, top_k)
    return _search_hybrid(database_url, question, vector, top_k)


def _search_vector(
    database_url: str, vector: list[float], top_k: int
) -> list[dict]:
    literal = to_vector_literal(vector)
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT c.text, d.filename, c.ord,"
            " 1 - (c.embedding <=> %s::vector) AS score"
            " FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE c.embedding IS NOT NULL"
            " ORDER BY c.embedding <=> %s::vector LIMIT %s",
            (literal, literal, top_k),
        ).fetchall()
    return [
        {
            "content": text,
            "doc_name": name,
            "chunk_index": ord_,
            "score": float(score),
            "mode": "vector",
        }
        for text, name, ord_, score in rows
    ]


def _search_hybrid(
    database_url: str, question: str, vector: list[float], top_k: int
) -> list[dict]:
    literal = to_vector_literal(vector)
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "WITH vec AS ("
            " SELECT id, ROW_NUMBER() OVER"
            " (ORDER BY embedding <=> %s::vector) AS rnk"
            " FROM chunks WHERE embedding IS NOT NULL"
            "), fts AS ("
            " SELECT id, ROW_NUMBER() OVER"
            " (ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC)"
            " AS rnk FROM chunks"
            " WHERE tsv @@ plainto_tsquery('english', %s)"
            ") SELECT c.text, d.filename, c.ord,"
            f" COALESCE(1.0/({RRF_K}+vec.rnk),0)"
            f" + COALESCE(1.0/({RRF_K}+fts.rnk),0) AS score"
            " FROM chunks c JOIN documents d ON d.id = c.document_id"
            " LEFT JOIN vec ON vec.id = c.id"
            " LEFT JOIN fts ON fts.id = c.id"
            " WHERE vec.id IS NOT NULL OR fts.id IS NOT NULL"
            " ORDER BY score DESC LIMIT %s",
            (literal, question, question, top_k),
        ).fetchall()
    return [
        {
            "content": text,
            "doc_name": name,
            "chunk_index": ord_,
            "score": float(score),
            "mode": "hybrid",
        }
        for text, name, ord_, score in rows
    ]
