"""Cache tests: all five layers (offline stubs; redis if reachable)."""

from __future__ import annotations

import math
import os

import psycopg
import pytest
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from rag import cache as rag_cache
from rag.embeddings import batch_embed
from rag.ingest import get_corpus_version, ingest_file
from rag.models import EMBEDDING_DIM
from rag.nim_client import NimClient

MODEL = "stub-model"


class CacheStub:
    def __init__(self, replies=None):
        self.chat_model = GenericFakeChatModel(
            messages=iter([AIMessage(content=r) for r in (replies or [])])
        )
        self.embed_calls = 0

    def _vec(self, text):
        low = text.lower()
        slots = set()
        if "alpha" in low:
            slots.add(0)
        if "beta" in low:
            slots.add(EMBEDDING_DIM - 1)
        v = [0.0] * EMBEDDING_DIM
        if not slots:
            v[1] = 1.0
            return v
        w = 1.0 / math.sqrt(len(slots))
        for s in slots:
            v[s] = w
        return v

    def embed(self, texts, input_type="passage"):
        self.embed_calls += 1
        return [self._vec(t) for t in texts]

    def chat(self, messages):
        return "rag"


@pytest.fixture(scope="module")
def cache_corpus(migrated_db):
    stub = CacheStub(replies=["CANNED"] * 50)
    res = ingest_file(
        b"alpha cache marker content",
        "t007-alpha.txt",
        database_url=migrated_db,
        client=stub,
        embed_model=MODEL,
    )
    yield {"url": migrated_db, "stub": stub, "doc_id": res["document_id"]}
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        conn.execute(
            "DELETE FROM documents WHERE id = %s", (res["document_id"],)
        )
        conn.execute("DELETE FROM embedding_cache")
        conn.execute("DELETE FROM answer_cache")
        conn.execute("DELETE FROM tool_cache")


@pytest.fixture
def fresh_caches(cache_corpus):
    url = cache_corpus["url"]
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DELETE FROM embedding_cache")
        conn.execute("DELETE FROM answer_cache")
        conn.execute("DELETE FROM tool_cache")
    return cache_corpus


def test_norm_and_exact_key():
    assert rag_cache.norm_question("  Hello   WORLD \n") == "hello world"
    k1 = rag_cache.exact_key("Hello", 3, MODEL, {"system": 1, "answer": 1})
    k2 = rag_cache.exact_key("hello ", 3, MODEL, {"system": 1, "answer": 1})
    assert k1 == k2  # normalization-insensitive
    assert rag_cache.exact_key("Hello", 4, MODEL, {"system": 1, "answer": 1}) != k1
    assert rag_cache.exact_key("Hello", 3, "other", {"system": 1, "answer": 1}) != k1
    assert rag_cache.exact_key("Hello", 3, MODEL, {"system": 1, "answer": 2}) != k1


def test_embedding_cache_avoids_repeated_calls(fresh_caches):
    url = fresh_caches["url"]
    stub = CacheStub()
    first = batch_embed(
        stub, ["alpha one", "beta two"], database_url=url, embed_model=MODEL
    )
    assert stub.embed_calls == 1
    second = batch_embed(
        stub, ["alpha one", "beta two"], database_url=url, embed_model=MODEL
    )
    assert stub.embed_calls == 1  # fully cached
    assert first == second
    batch_embed(
        stub, ["alpha one", "gamma three"],
        database_url=url, embed_model=MODEL,
    )
    assert stub.embed_calls == 2  # only the new text embedded


def test_repeat_question_is_exact_hit(fresh_caches):
    url = fresh_caches["url"]
    stub = CacheStub(replies=["CANNED"])
    args = {
        "database_url": url,
        "client": stub,
        "chat_model": "stub-chat",
        "embed_model": MODEL,
    }
    first = rag_cache.cached_ask("alpha query", **args)
    assert first["cache_hit"] is False
    assert first["cache_layer"] is None
    second = rag_cache.cached_ask("alpha query", **args)
    assert second["cache_hit"] is True
    assert second["cache_layer"] == "exact"
    assert second["answer"] == first["answer"]  # 1 LLM call total


def test_paraphrase_is_semantic_hit(fresh_caches):
    url = fresh_caches["url"]
    stub = CacheStub(replies=["CANNED"])
    args = {
        "database_url": url,
        "client": stub,
        "chat_model": "stub-chat",
        "embed_model": MODEL,
    }
    rag_cache.cached_ask("alpha query", **args)
    # Different normalized text, identical embedding (sim 1.0 >= 0.95).
    res = rag_cache.cached_ask("alpha query!", **args)
    assert res["cache_hit"] is True
    assert res["cache_layer"] == "semantic"


def test_below_threshold_paraphrase_misses(fresh_caches):
    url = fresh_caches["url"]
    stub = CacheStub(replies=["CANNED", "CANNED2"])
    args = {
        "database_url": url,
        "client": stub,
        "chat_model": "stub-chat",
        "embed_model": MODEL,
    }
    rag_cache.cached_ask("alpha query", **args)
    # sim("alpha beta", "alpha") = 0.707 < 0.95 -> miss, new LLM call.
    res = rag_cache.cached_ask("alpha beta query", **args)
    assert res["cache_hit"] is False
    assert "CANNED2" in res["answer"]


def test_new_upload_invalidates_answer_cache(fresh_caches):
    url = fresh_caches["url"]
    stub = CacheStub(replies=["CANNED", "CANNED2"])
    args = {
        "database_url": url,
        "client": stub,
        "chat_model": "stub-chat",
        "embed_model": MODEL,
    }
    rag_cache.cached_ask("alpha query", **args)
    assert rag_cache.cached_ask("alpha query", **args)["cache_hit"] is True
    ingest_file(
        b"brand new t007 invalidation doc",
        "t007-new.txt",
        database_url=url,
        client=stub,
        embed_model=MODEL,
    )
    try:
        res = rag_cache.cached_ask("alpha query", **args)
        assert res["cache_hit"] is False
        assert "CANNED2" in res["answer"]
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM documents WHERE filename = 't007-new.txt'"
            )


def test_clear_caches_empties_all(fresh_caches):
    url = fresh_caches["url"]
    stub = CacheStub(replies=["CANNED"])
    rag_cache.cached_ask(
        "alpha query", database_url=url, client=stub,
        chat_model="stub-chat", embed_model=MODEL,
    )
    rag_cache.tool_set(url, "k", {"v": 1})
    counts = rag_cache.clear_caches(url)
    assert counts["embedding_cache"] >= 1
    assert counts["answer_cache"] == 1
    assert counts["tool_cache"] == 1
    with psycopg.connect(url) as conn:
        for table in ("embedding_cache", "answer_cache", "tool_cache"):
            assert (
                conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                == 0
            )


def test_tool_cache_postgres_roundtrip_and_expiry(fresh_caches):
    url = fresh_caches["url"]
    assert rag_cache.tool_get(url, "missing") is None
    rag_cache.tool_set(url, "city:pune", {"temp": 28})
    assert rag_cache.tool_get(url, "city:pune") == {"temp": 28}
    rag_cache.tool_set(url, "old", {"x": 1}, ttl_seconds=-1)
    assert rag_cache.tool_get(url, "old") is None


def test_tool_cache_redis_backend_if_available(fresh_caches):
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        pytest.skip("REDIS_URL not set; postgres fallback covered above")
    try:
        rag_cache.tool_set(
            fresh_caches["url"], "rk", {"v": 2}, redis_url=redis_url
        )
    except Exception as exc:  # noqa: BLE001 - any redis error skips
        pytest.skip(f"redis unreachable: {type(exc).__name__}")
    assert rag_cache.tool_get(
        fresh_caches["url"], "rk", redis_url=redis_url
    ) == {"v": 2}
    rag_cache.clear_caches(fresh_caches["url"], redis_url=redis_url)
    assert rag_cache.tool_get(
        fresh_caches["url"], "rk", redis_url=redis_url
    ) is None


def test_ask_with_history_uses_cache_when_enabled(fresh_caches):
    from rag import chat as rag_chat

    url = fresh_caches["url"]
    stub = CacheStub(replies=["CANNED"])
    first = rag_chat.ask_with_history(
        "alpha query", database_url=url, client=stub,
        use_cache=True, chat_model_name="stub-chat", embed_model_name=MODEL,
    )
    assert first["cache_hit"] is False
    second = rag_chat.ask_with_history(
        "alpha query", database_url=url, client=stub,
        conversation_id=first["conversation_id"],
        use_cache=True, chat_model_name="stub-chat", embed_model_name=MODEL,
    )
    assert second["cache_hit"] is True
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            "DELETE FROM conversations WHERE id = %s",
            (first["conversation_id"],),
        )


@pytest.mark.live
def test_live_repeat_is_cache_hit(migrated_db):
    key = os.environ.get("NVIDIA_API_KEY")
    chat_model = os.environ.get("NIM_CHAT_MODEL")
    embed_model = os.environ.get("NIM_EMBED_MODEL")
    if not (key and chat_model and embed_model):
        pytest.skip("needs NVIDIA_API_KEY + NIM_CHAT_MODEL + NIM_EMBED_MODEL")
    client = NimClient(chat_model=chat_model, embed_model=embed_model)
    doc_ids = []
    try:
        for name in ("rivers.txt", "curie.txt"):
            from pathlib import Path

            res = ingest_file(
                (Path("tests/fixtures") / name).read_bytes(),
                f"t007live-{name}",
                database_url=migrated_db,
                client=client,
                embed_model=embed_model,
            )
            doc_ids.append(res["document_id"])
        args = {
            "database_url": migrated_db,
            "client": client,
            "chat_model": chat_model,
            "embed_model": embed_model,
        }
        first = rag_cache.cached_ask("Which river is the longest?", **args)
        assert first["cache_hit"] is False
        assert "Nile" in first["answer"]
        second = rag_cache.cached_ask("Which river is the longest?", **args)
        assert second["cache_hit"] is True
        assert second["answer"] == first["answer"]
        assert get_corpus_version(migrated_db) >= 1
    finally:
        with psycopg.connect(migrated_db, autocommit=True) as conn:
            if doc_ids:
                conn.execute(
                    "DELETE FROM documents WHERE id = ANY(%s::bigint[])",
                    (doc_ids,),
                )
            conn.execute("DELETE FROM answer_cache")
            conn.execute("DELETE FROM embedding_cache")
