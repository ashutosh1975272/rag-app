"""One-shot RAG answering: retrieve → threshold gate → cited answer.

The refusal gate always uses vector cosine similarity (uniform semantics in
both modes); context comes from the requested mode. Sources are appended
programmatically as ``[doc name #chunk]`` — never trusted to the model.
"""

from __future__ import annotations

import time

from rag import retrieve
from rag.llm import build_answer_chain
from rag.prompts import get_active, seed_defaults

REFUSAL = "I don't know based on the uploaded documents."

# Measured 2026-10-06 on the 3-doc fixture: in-doc top-1 0.45–0.66,
# out-of-doc top-1 0.01–0.07. 0.3 sits cleanly between; configurable.
DEFAULT_THRESHOLD = 0.3


def ask(
    question: str,
    *,
    database_url: str,
    client,
    top_k: int = 6,
    mode: str = "hybrid",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Answer *question* from the corpus, or refuse honestly.

    Returns ``{answer, sources, refused, prompt_versions, route,
    latency_ms, cache_hit}``. ``sources`` is ``[{doc_name, chunk_index,
    score}]``. Refusals make no LLM call.
    """
    start = time.monotonic()
    seed_defaults(database_url)
    vec_hits = retrieve.search(
        question, database_url=database_url, client=client,
        top_k=top_k, mode="vector",
    )
    if not vec_hits or vec_hits[0]["score"] < threshold:
        return {
            "answer": REFUSAL,
            "sources": [],
            "refused": True,
            "prompt_versions": {},
            "route": "rag",
            "latency_ms": _elapsed_ms(start),
            "cache_hit": False,
        }
    ctx_hits = (
        vec_hits
        if mode == "vector"
        else retrieve.search(
            question, database_url=database_url, client=client,
            top_k=top_k, mode=mode,
        )
    )
    sys_version, sys_text = get_active(database_url, "system")
    ans_version, ans_text = get_active(database_url, "answer")
    context = "\n\n".join(
        f"[{h['doc_name']} #{h['chunk_index']}]\n{h['content']}"
        for h in ctx_hits
    )
    chain = build_answer_chain(client.chat_model, sys_text, ans_text)
    text = chain.invoke({"context": context, "question": question})
    sources = [
        {
            "doc_name": h["doc_name"],
            "chunk_index": h["chunk_index"],
            "score": h["score"],
        }
        for h in ctx_hits
    ]
    cites = ", ".join(
        f"[{s['doc_name']} #{s['chunk_index']}]" for s in sources
    )
    return {
        "answer": f"{text}\n\nSources: {cites}",
        "sources": sources,
        "refused": False,
        "prompt_versions": {"system": sys_version, "answer": ans_version},
        "route": "rag",
        "latency_ms": _elapsed_ms(start),
        "cache_hit": False,
    }


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
