"""Retrieval tests: vector + hybrid ranking on the 3-doc fixture.

Stub embeddings are one-hot by topic marker (no network); live NIM tests
are marked `live` and assert the ticket's top-3 criterion.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest

from rag.ingest import ingest_file
from rag.models import EMBEDDING_DIM
from rag.nim_client import NimClient
from rag.retrieve import search

FIXTURES = Path(__file__).parent / "fixtures"

MARKERS = (
    (("nile", "amazon", "river", "rivers", "mediterranean"), 0),
    (("curie", "nobel", "radium", "polonium", "paris"), 1),
    (("bridge", "bridges", "gate", "millau"), 2),
)


class TopicStub:
    """One-hot by topic marker; unmarked text -> neutral slot 3."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * EMBEDDING_DIM
        low = text.lower()
        for words, slot in MARKERS:
            if any(w in low for w in words):
                v[slot] = 1.0
                return v
        v[3] = 1.0
        return v

    def embed(self, texts, input_type="passage"):
        return [self._vec(t) for t in texts]


@pytest.fixture(scope="module")
def stub_corpus(migrated_db):
    ids = []
    stub = TopicStub()
    for name in ("rivers.txt", "curie.txt", "bridges.txt"):
        res = ingest_file(
            (FIXTURES / name).read_bytes(),
            f"t004-{name}",
            database_url=migrated_db,
            client=stub,
        )
        ids.append(res["document_id"])
    yield {"url": migrated_db, "stub": stub, "ids": ids}
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        conn.execute(
            "DELETE FROM documents WHERE id = ANY(%s::bigint[])", (ids,)
        )


def _top_names(hits, k=1):
    return [h["doc_name"] for h in hits[:k]]


def test_vector_top1_is_known_chunk(stub_corpus):
    hits = search(
        "Which river is the longest in Africa?",
        database_url=stub_corpus["url"],
        client=stub_corpus["stub"],
        top_k=3,
        mode="vector",
    )
    assert len(hits) == 3
    assert hits[0]["doc_name"] == "t004-rivers.txt"
    assert "Nile" in hits[0]["content"]
    assert hits[0]["mode"] == "vector"
    assert set(hits[0]) == {
        "content",
        "doc_name",
        "chunk_index",
        "score",
        "mode",
    }
    assert hits[0]["score"] == pytest.approx(1.0)


def test_hybrid_keyword_query_finds_bridges(stub_corpus):
    question = "tallest viaduct 343 meters"
    hybrid = search(
        question,
        database_url=stub_corpus["url"],
        client=stub_corpus["stub"],
        top_k=3,
        mode="hybrid",
    )
    assert hybrid[0]["doc_name"] == "t004-bridges.txt"
    assert hybrid[0]["mode"] == "hybrid"
    vector = search(
        question,
        database_url=stub_corpus["url"],
        client=stub_corpus["stub"],
        top_k=3,
        mode="vector",
    )
    print(
        "\nvector top-3:",
        _top_names(vector, 3),
        "| hybrid top-3:",
        _top_names(hybrid, 3),
    )


def test_empty_db_returns_empty_list(empty_db_url):
    assert (
        search(
            "anything at all",
            database_url=empty_db_url,
            client=TopicStub(),
            mode="vector",
        )
        == []
    )
    assert (
        search(
            "anything at all",
            database_url=empty_db_url,
            client=TopicStub(),
            mode="hybrid",
        )
        == []
    )


def test_blank_question_returns_empty_list(stub_corpus):
    assert (
        search(
            "   ",
            database_url=stub_corpus["url"],
            client=stub_corpus["stub"],
        )
        == []
    )


def test_invalid_mode_rejected(stub_corpus):
    with pytest.raises(ValueError, match="mode"):
        search(
            "q",
            database_url=stub_corpus["url"],
            client=stub_corpus["stub"],
            mode="sideways",
        )


def _live_client():
    key = os.environ.get("NVIDIA_API_KEY")
    chat_model = os.environ.get("NIM_CHAT_MODEL")
    embed_model = os.environ.get("NIM_EMBED_MODEL")
    if not (key and chat_model and embed_model):
        pytest.skip("needs NVIDIA_API_KEY + NIM_CHAT_MODEL + NIM_EMBED_MODEL")
    return NimClient(chat_model=chat_model, embed_model=embed_model)


@pytest.mark.live
def test_live_vector_top3_has_answer(migrated_db):
    client = _live_client()
    ids = []
    try:
        for name in ("rivers.txt", "curie.txt", "bridges.txt"):
            res = ingest_file(
                (FIXTURES / name).read_bytes(),
                f"t004live-{name}",
                database_url=migrated_db,
                client=client,
            )
            ids.append(res["document_id"])
        hits = search(
            "Which river is the longest in Africa?",
            database_url=migrated_db,
            client=client,
            top_k=3,
            mode="vector",
        )
        print("\nlive vector top-3:", _top_names(hits, 3))
        assert any("Nile" in h["content"] for h in hits)
    finally:
        if ids:
            with psycopg.connect(migrated_db, autocommit=True) as conn:
                conn.execute(
                    "DELETE FROM documents WHERE id = ANY(%s::bigint[])",
                    (ids,),
                )


@pytest.mark.live
def test_live_hybrid_keyword_top3(migrated_db):
    client = _live_client()
    ids = []
    try:
        for name in ("rivers.txt", "curie.txt", "bridges.txt"):
            res = ingest_file(
                (FIXTURES / name).read_bytes(),
                f"t004live-{name}",
                database_url=migrated_db,
                client=client,
            )
            ids.append(res["document_id"])
        hits = search(
            "Millau Viaduct height in meters",
            database_url=migrated_db,
            client=client,
            top_k=3,
            mode="hybrid",
        )
        print("\nlive hybrid top-3:", _top_names(hits, 3))
        assert any("Millau" in h["content"] for h in hits)
    finally:
        if ids:
            with psycopg.connect(migrated_db, autocommit=True) as conn:
                conn.execute(
                    "DELETE FROM documents WHERE id = ANY(%s::bigint[])",
                    (ids,),
                )
