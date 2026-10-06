"""Answering tests: prompts in DB, citations, refusal gate.

Stub embeddings (alpha/beta one-hots) plus a fake chat model keep these
offline; live NIM tests are marked `live`.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import psycopg
import pytest
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from rag.chat import REFUSAL, ask
from rag.ingest import ingest_file
from rag.llm import build_answer_chain
from rag.models import EMBEDDING_DIM
from rag.nim_client import NimClient
from rag.prompts import PromptError, add_version, get_active, seed_defaults

FIXTURES = Path(__file__).parent / "fixtures"
CITATION = re.compile(r"\[.+? #\d+\]")


class AnswerStub:
    """alpha->e0, beta->e2047 (normalized on multi-match), else e1."""

    def __init__(self, replies=None):
        self.chat_model = GenericFakeChatModel(
            messages=iter([AIMessage(content=r) for r in (replies or [])])
        )

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


@pytest.fixture(scope="module")
def answer_corpus(migrated_db):
    stub = AnswerStub(replies=["CANNED"] * 50)
    ids = []
    for name, text in (
        ("t005-alpha.txt", "alpha marker content sunny day"),
        ("t005-beta.txt", "beta marker content rainy night"),
    ):
        res = ingest_file(
            text.encode(), name, database_url=migrated_db, client=stub
        )
        ids.append(res["document_id"])
    seed_defaults(migrated_db)
    yield {"url": migrated_db, "stub": stub, "ids": ids}
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        conn.execute(
            "DELETE FROM documents WHERE id = ANY(%s::bigint[])", (ids,)
        )


def test_seed_is_idempotent_and_active(answer_corpus):
    url = answer_corpus["url"]
    seed_defaults(url)
    seed_defaults(url)
    for name in ("system", "answer", "query-rewrite", "router"):
        version, content = get_active(url, name)
        assert version == 1
        assert content.strip()


def test_get_active_missing_raises(empty_db_url):
    with psycopg.connect(empty_db_url, autocommit=True) as conn:
        conn.execute("DELETE FROM prompt_templates")
    with pytest.raises(PromptError, match="no-such-prompt"):
        get_active(empty_db_url, "no-such-prompt")


def test_add_version_activates_new(answer_corpus):
    url = answer_corpus["url"]
    version, _ = get_active(url, "answer")
    new = add_version(url, "answer", "v-new {context} {question}")
    assert new == version + 1
    got_version, got_content = get_active(url, "answer")
    assert got_version == new
    assert got_content.startswith("v-new")


def test_in_doc_answer_has_citation_and_versions(answer_corpus):
    url = answer_corpus["url"]
    add_version(url, "answer", "Answer: {question}\nContext: {context}")
    res = ask(
        "alpha query",
        database_url=url,
        client=answer_corpus["stub"],
        mode="vector",
    )
    assert res["refused"] is False
    assert "CANNED" in res["answer"]
    cites = CITATION.findall(res["answer"])
    assert len(cites) >= 1
    assert res["sources"]
    assert res["sources"][0]["doc_name"] == "t005-alpha.txt"
    assert res["prompt_versions"]["system"] == 1
    assert res["prompt_versions"]["answer"] >= 2
    assert res["route"] == "rag"
    assert res["cache_hit"] is False
    assert isinstance(res["latency_ms"], int)


def test_all_citations_resolve_to_sources(answer_corpus):
    res = ask(
        "alpha query",
        database_url=answer_corpus["url"],
        client=answer_corpus["stub"],
        mode="vector",
    )
    known = {
        f"[{s['doc_name']} #{s['chunk_index']}]" for s in res["sources"]
    }
    for cite in CITATION.findall(res["answer"]):
        assert cite in known


def test_out_of_doc_refuses_exact_string(answer_corpus):
    res = ask(
        "zzz nothing here",
        database_url=answer_corpus["url"],
        client=answer_corpus["stub"],
        mode="vector",
    )
    assert res["refused"] is True
    assert res["answer"] == REFUSAL
    assert res["sources"] == []


def test_threshold_gate_forces_refusal(answer_corpus):
    url = answer_corpus["url"]
    ok = ask(
        "alpha beta query",
        database_url=url,
        client=answer_corpus["stub"],
        mode="vector",
    )
    assert ok["refused"] is False  # 0.707 >= default 0.3
    forced = ask(
        "alpha beta query",
        database_url=url,
        client=answer_corpus["stub"],
        mode="vector",
        threshold=1.0,
    )
    assert forced["refused"] is True
    assert forced["answer"] == REFUSAL


def test_empty_corpus_refuses_without_llm_call(empty_db_url):
    stub = AnswerStub(replies=[])  # any LLM call raises StopIteration
    res = ask("alpha query", database_url=empty_db_url, client=stub)
    assert res["refused"] is True
    assert res["answer"] == REFUSAL


def test_answer_chain_formats_prompt():
    seen = []

    class Capture(GenericFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.extend(messages)
            return super()._generate(messages, stop, run_manager, **kwargs)

    chat = Capture(messages=iter([AIMessage(content="done")]))
    chain = build_answer_chain(chat, "SYS", "Q:{question} C:{context}")
    assert chain.invoke({"question": "q1", "context": "c1"}) == "done"
    assert seen[0].content == "SYS"
    assert seen[1].content == "Q:q1 C:c1"


def _live_client():
    key = os.environ.get("NVIDIA_API_KEY")
    chat_model = os.environ.get("NIM_CHAT_MODEL")
    embed_model = os.environ.get("NIM_EMBED_MODEL")
    if not (key and chat_model and embed_model):
        pytest.skip("needs NVIDIA_API_KEY + NIM_CHAT_MODEL + NIM_EMBED_MODEL")
    return NimClient(chat_model=chat_model, embed_model=embed_model)


@pytest.mark.live
def test_live_in_doc_cited_and_out_of_doc_refused(migrated_db):
    client = _live_client()
    ids = []
    try:
        for name in ("rivers.txt", "curie.txt", "bridges.txt"):
            res = ingest_file(
                (FIXTURES / name).read_bytes(),
                f"t005live-{name}",
                database_url=migrated_db,
                client=client,
            )
            ids.append(res["document_id"])
        cited = ask(
            "Which river is the longest in Africa?",
            database_url=migrated_db,
            client=client,
        )
        assert cited["refused"] is False
        assert "Nile" in cited["answer"]
        assert CITATION.search(cited["answer"])
        refused = ask(
            "What is the capital of Atlantis?",
            database_url=migrated_db,
            client=client,
        )
        assert refused["refused"] is True
        assert refused["answer"] == REFUSAL
    finally:
        if ids:
            with psycopg.connect(migrated_db, autocommit=True) as conn:
                conn.execute(
                    "DELETE FROM documents WHERE id = ANY(%s::bigint[])",
                    (ids,),
                )
