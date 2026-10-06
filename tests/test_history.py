"""History tests: conversations, rewrite, feedback, export (offline stubs)."""

from __future__ import annotations

import math
import os

import psycopg
import pytest
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from rag import chat as rag_chat
from rag.ingest import ingest_file
from rag.models import EMBEDDING_DIM
from rag.nim_client import NimClient


class HistoryStub:
    def __init__(self, replies=None, rewrite_fn=None):
        self.chat_model = GenericFakeChatModel(
            messages=iter([AIMessage(content=r) for r in (replies or [])])
        )
        self._rewrite_fn = rewrite_fn or (lambda q, h: f"RW:{q}")
        self.rewrite_calls = []

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
        return [self._vec(t) for t in texts]

    def chat(self, messages):
        self.rewrite_calls.append(messages)
        return self._rewrite_fn(messages[-1]["content"], messages)


@pytest.fixture(scope="module")
def hist_corpus(migrated_db):
    stub = HistoryStub(replies=["CANNED"] * 50)
    res = ingest_file(
        b"alpha history marker content",
        "t006-alpha.txt",
        database_url=migrated_db,
        client=stub,
    )
    yield {"url": migrated_db, "stub": stub, "doc_id": res["document_id"]}
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        conn.execute(
            "DELETE FROM documents WHERE id = %s", (res["document_id"],)
        )


@pytest.fixture
def conv_ids(hist_corpus):
    ids: list[int] = []
    yield (hist_corpus["url"], hist_corpus["stub"], ids)
    if ids:
        with psycopg.connect(hist_corpus["url"], autocommit=True) as conn:
            conn.execute(
                "DELETE FROM conversations WHERE id = ANY(%s::bigint[])",
                (ids,),
            )


def test_create_save_list_roundtrip(conv_ids):
    url, _, ids = conv_ids
    conv = rag_chat.create_conversation(url, title="t006-roundtrip")
    ids.append(conv["id"])
    rag_chat.save_message(url, conv["id"], "user", "hello alpha")
    rag_chat.save_message(
        url, conv["id"], "assistant", "hi back",
        sources=[{"doc_name": "d", "chunk_index": 0, "score": 1.0}],
        route="rag", latency_ms=12, cache_hit=False,
    )
    got = rag_chat.get_conversation(url, conv["id"])
    assert got["title"] == "t006-roundtrip"
    assert [m["role"] for m in got["messages"]] == ["user", "assistant"]
    assert got["messages"][1]["route"] == "rag"
    assert got["messages"][1]["latency_ms"] == 12
    assert got["messages"][1]["cache_hit"] is False
    listed = rag_chat.list_conversations(url)
    row = next(c for c in listed if c["id"] == conv["id"])
    assert row["message_count"] == 2


def test_rename_and_delete(conv_ids):
    url, _, ids = conv_ids
    conv = rag_chat.create_conversation(url, title="t006-temp")
    ids.append(conv["id"])
    rag_chat.save_message(url, conv["id"], "user", "x")
    rag_chat.rename_conversation(url, conv["id"], "t006-renamed")
    assert rag_chat.get_conversation(url, conv["id"])["title"] == (
        "t006-renamed"
    )
    rag_chat.delete_conversation(url, conv["id"])
    ids.remove(conv["id"])
    with pytest.raises(rag_chat.HistoryError, match="conversation"):
        rag_chat.get_conversation(url, conv["id"])


def test_rewrite_resolves_followup(hist_corpus):
    stub = HistoryStub(
        rewrite_fn=lambda q, h: f"Standalone about Nile: {q}",
    )
    out = rag_chat.rewrite_followup(
        "what about the second one?",
        [{"role": "user", "content": "Which river is longest?"},
         {"role": "assistant", "content": "The Nile is the longest."}],
        database_url=hist_corpus["url"],
        client=stub,
    )
    assert out["rewritten"] is True
    assert "Nile" in out["question"]  # antecedent carried over
    assert len(stub.rewrite_calls) == 1


def test_rewrite_skipped_for_self_contained(hist_corpus):
    stub = hist_corpus["stub"]
    before = len(stub.rewrite_calls)
    out = rag_chat.rewrite_followup(
        "What is the boiling point of water at sea level?",
        [],
        database_url=hist_corpus["url"],
        client=stub,
    )
    assert out == {
        "question": "What is the boiling point of water at sea level?",
        "rewritten": False,
    }
    assert len(stub.rewrite_calls) == before  # no LLM call


def test_feedback_persists_and_validates(conv_ids):
    url, _, ids = conv_ids
    conv = rag_chat.create_conversation(url, title="t006-fb")
    ids.append(conv["id"])
    mid = rag_chat.save_message(url, conv["id"], "assistant", "ans")
    rag_chat.set_feedback(url, mid, 1)
    got = rag_chat.get_conversation(url, conv["id"])
    assert got["messages"][0]["feedback"] == 1
    rag_chat.set_feedback(url, mid, -1)
    assert rag_chat.get_conversation(url, conv["id"])["messages"][0][
        "feedback"
    ] == -1
    with pytest.raises(rag_chat.HistoryError, match="feedback"):
        rag_chat.set_feedback(url, mid, 5)


def test_export_markdown_contains_all(conv_ids):
    url, _, ids = conv_ids
    conv = rag_chat.create_conversation(url, title="t006-export")
    ids.append(conv["id"])
    rag_chat.save_message(url, conv["id"], "user", "alpha query")
    rag_chat.save_message(
        url, conv["id"], "assistant", "alpha answer [t006-alpha.txt #0]",
        sources=[{"doc_name": "t006-alpha.txt", "chunk_index": 0,
                  "score": 0.9}],
        route="rag", latency_ms=7, cache_hit=True,
    )
    md = rag_chat.export_markdown(url, conv["id"])
    assert "# t006-export" in md
    assert "alpha query" in md
    assert "alpha answer" in md
    assert "[t006-alpha.txt #0]" in md


def test_save_message_validates_role(conv_ids):
    url, _, ids = conv_ids
    conv = rag_chat.create_conversation(url, title="t006-role")
    ids.append(conv["id"])
    with pytest.raises(rag_chat.HistoryError, match="role"):
        rag_chat.save_message(url, conv["id"], "pirate", "arr")


def test_ask_with_history_two_turns(conv_ids):
    url, stub, ids = conv_ids
    first = rag_chat.ask_with_history(
        "alpha query", database_url=url, client=stub, conversation_id=None
    )
    ids.append(first["conversation_id"])
    assert first["refused"] is False
    assert first["rewritten"] is False
    conv = rag_chat.get_conversation(url, first["conversation_id"])
    assert conv["title"] == "alpha query"
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][0]["content"] == "alpha query"
    assert conv["messages"][1]["route"] == "rag"
    second = rag_chat.ask_with_history(
        "what about beta?",
        database_url=url,
        client=stub,
        conversation_id=first["conversation_id"],
    )
    assert second["rewritten"] is True
    assert len(rag_chat.get_conversation(url, first["conversation_id"])[
        "messages"
    ]) == 4


@pytest.mark.live
def test_live_rewrite_contains_antecedent(migrated_db):
    key = os.environ.get("NVIDIA_API_KEY")
    chat_model = os.environ.get("NIM_CHAT_MODEL")
    embed_model = os.environ.get("NIM_EMBED_MODEL")
    if not (key and chat_model and embed_model):
        pytest.skip("needs NVIDIA_API_KEY + NIM_CHAT_MODEL + NIM_EMBED_MODEL")
    client = NimClient(chat_model=chat_model, embed_model=embed_model)
    out = rag_chat.rewrite_followup(
        "what about the second one?",
        [{"role": "user", "content": "Which river is the longest?"},
         {"role": "assistant",
          "content": "The Nile is the longest river in Africa."}],
        database_url=migrated_db,
        client=client,
    )
    assert out["rewritten"] is True
    assert "river" in out["question"].lower() or "Nile" in out["question"]
