"""One-shot RAG answering: retrieve → threshold gate → cited answer.

The refusal gate always uses vector cosine similarity (uniform semantics in
both modes); context comes from the requested mode. Sources are appended
programmatically as ``[doc name #chunk]`` — never trusted to the model.

Also conversation history: CRUD, follow-up rewrite, feedback, export, and
`ask_with_history()` turns that persist every message with route/latency.
"""

from __future__ import annotations

import json
import logging
import re
import time

import psycopg

from rag import retrieve
from rag.llm import build_answer_chain
from rag.prompts import get_active, seed_defaults
from rag.router import route as route_message
from rag.tools_weather import UnknownCity, WeatherError, get_weather

logger = logging.getLogger(__name__)

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


class HistoryError(ValueError):
    """Conversation/history operation failed."""


VALID_ROLES = ("user", "assistant", "system")

FOLLOWUP_PRONOUNS = frozenset(
    [
        "it", "its", "they", "them", "their", "theirs",
        "this", "that", "these", "those", "he", "him",
        "his", "she", "her", "hers", "one", "ones",
    ]
)
FOLLOWUP_PREFIXES = ("what about", "how about", "and ", "then ")
FOLLOWUP_MAX_WORDS = 14


def create_conversation(database_url: str, title: str = "New chat") -> dict:
    with psycopg.connect(database_url, autocommit=True) as conn:
        row = conn.execute(
            "INSERT INTO conversations (title)"
            " VALUES (%s) RETURNING id, title",
            (title or "New chat",),
        ).fetchone()
        assert row is not None
    return {"id": row[0], "title": row[1]}


def list_conversations(database_url: str) -> list[dict]:
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at,"
            " COUNT(m.id) AS message_count"
            " FROM conversations c LEFT JOIN messages m"
            " ON m.conversation_id = c.id"
            " GROUP BY c.id ORDER BY c.updated_at DESC, c.id DESC"
        ).fetchall()
    return [
        {
            "id": cid,
            "title": title,
            "created_at": created.isoformat() if created else None,
            "updated_at": updated.isoformat() if updated else None,
            "message_count": count,
        }
        for cid, title, created, updated, count in rows
    ]


def get_conversation(database_url: str, conversation_id: int) -> dict:
    with psycopg.connect(database_url) as conn:
        conv = conn.execute(
            "SELECT id, title FROM conversations WHERE id = %s",
            (conversation_id,),
        ).fetchone()
        if conv is None:
            raise HistoryError(f"Unknown conversation: {conversation_id}")
        rows = conn.execute(
            "SELECT id, role, content, sources, route, latency_ms,"
            " cache_hit, feedback, created_at FROM messages"
            " WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return {
        "id": conv[0],
        "title": conv[1],
        "messages": [
            {
                "id": mid,
                "role": role,
                "content": content,
                "sources": sources or [],
                "route": route,
                "latency_ms": latency,
                "cache_hit": cache_hit,
                "feedback": feedback,
                "created_at": created.isoformat() if created else None,
            }
            for mid, role, content, sources, route, latency, cache_hit,
            feedback, created in rows
        ],
    }


def rename_conversation(
    database_url: str, conversation_id: int, title: str
) -> None:
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(
            "UPDATE conversations SET title = %s, updated_at = now()"
            " WHERE id = %s",
            (title, conversation_id),
        )


def delete_conversation(database_url: str, conversation_id: int) -> None:
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(
            "DELETE FROM conversations WHERE id = %s", (conversation_id,)
        )


def save_message(
    database_url: str,
    conversation_id: int,
    role: str,
    content: str,
    *,
    sources: list | None = None,
    route: str | None = None,
    latency_ms: int | None = None,
    cache_hit: bool = False,
) -> int:
    if role not in VALID_ROLES:
        raise HistoryError(
            f"role must be one of {VALID_ROLES}, got {role!r}"
        )
    with psycopg.connect(database_url, autocommit=True) as conn:
        row = conn.execute(
            "INSERT INTO messages"
            " (conversation_id, role, content, sources, route,"
            " latency_ms, cache_hit)"
            " VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s) RETURNING id",
            (
                conversation_id,
                role,
                content,
                json.dumps(sources or []),
                route,
                latency_ms,
                cache_hit,
            ),
        ).fetchone()
        assert row is not None
        conn.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = %s",
            (conversation_id,),
        )
    return row[0]


def set_feedback(
    database_url: str, message_id: int, value: int | None
) -> None:
    if value not in (1, -1, None):
        raise HistoryError(f"feedback must be +1, -1, or None, got {value!r}")
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(
            "UPDATE messages SET feedback = %s WHERE id = %s",
            (value, message_id),
        )


def looks_like_followup(question: str) -> bool:
    """Heuristic: short + pronoun/follow-up prefix (logged by caller)."""
    words = re.findall(r"[a-zA-Z']+", question.lower())
    if not words or len(words) > FOLLOWUP_MAX_WORDS:
        return False
    if any(p in FOLLOWUP_PRONOUNS for p in words):
        return True
    return question.lower().lstrip().startswith(FOLLOWUP_PREFIXES)


def rewrite_followup(
    question: str,
    history: list,
    *,
    database_url: str,
    client,
) -> dict:
    """Rewrite a follow-up into a standalone question (or skip)."""
    if not looks_like_followup(question):
        logger.info("rewrite skipped (self-contained): %r", question[:60])
        return {"question": question, "rewritten": False}
    seed_defaults(database_url)
    _, template = get_active(database_url, "query-rewrite")
    turns = history[-4:] if len(history) > 4 else history
    history_text = "\n".join(
        f"{t['role']}: {t['content']}" for t in turns
    )
    standalone = client.chat(
        [
            {
                "role": "user",
                "content": template.format(
                    history=history_text, question=question
                ),
            }
        ]
    ).strip()
    logger.info("rewrite applied: %r -> %r", question[:60], standalone[:60])
    return {"question": standalone, "rewritten": True}


def export_markdown(database_url: str, conversation_id: int) -> str:
    conv = get_conversation(database_url, conversation_id)
    lines = [f"# {conv['title']}", ""]
    for msg in conv["messages"]:
        lines.append(f"## {msg['role']}")
        lines.append("")
        lines.append(msg["content"])
        if msg["role"] == "assistant" and msg["sources"]:
            cites = ", ".join(
                f"[{s['doc_name']} #{s['chunk_index']}]"
                for s in msg["sources"]
            )
            lines.append("")
            lines.append(f"Sources: {cites}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def ask_with_history(
    question: str,
    *,
    database_url: str,
    client,
    conversation_id: int | None = None,
    top_k: int = 6,
    mode: str = "hybrid",
    threshold: float = DEFAULT_THRESHOLD,
    use_cache: bool = False,
    chat_model_name: str | None = None,
    embed_model_name: str | None = None,
    semantic_threshold: float = 0.95,
    redis_url: str | None = None,
) -> dict:
    """One persisted turn: route → branch → save user+assistant rows."""
    start = time.monotonic()
    if conversation_id is None:
        conversation_id = create_conversation(database_url)["id"]
        fresh = True
    else:
        get_conversation(database_url, conversation_id)  # validates
        fresh = False
    routing = route_message(question, database_url=database_url, client=client)
    if routing["route"] == "weather":
        result = _weather_turn(
            question,
            routing.get("city"),
            database_url=database_url,
            redis_url=redis_url,
        )
        rewritten: dict = {"question": question, "rewritten": False}
    elif routing["route"] == "chitchat":
        text = client.chat(
            [
                {
                    "role": "system",
                    "content": "You are a friendly assistant. "
                    "Reply in one short sentence.",
                },
                {"role": "user", "content": question},
            ]
        ).strip()
        result = {
            "answer": text,
            "sources": [],
            "refused": False,
            "route": "chitchat",
            "latency_ms": _elapsed_ms(start),
            "cache_hit": False,
            "weather": None,
        }
        rewritten = {"question": question, "rewritten": False}
    else:
        history = get_conversation(database_url, conversation_id)["messages"]
        rewritten = rewrite_followup(
            question, history, database_url=database_url, client=client
        )
        if use_cache:
            from rag.cache import cached_ask  # lazy: cache imports ask

            if chat_model_name is None or embed_model_name is None:
                raise HistoryError(
                    "use_cache needs chat_model_name and embed_model_name"
                )
            result = cached_ask(
                rewritten["question"],
                database_url=database_url,
                client=client,
                chat_model=chat_model_name,
                embed_model=embed_model_name,
                top_k=top_k,
                mode=mode,
                threshold=threshold,
                semantic_threshold=semantic_threshold,
            )
        else:
            result = ask(
                rewritten["question"],
                database_url=database_url,
                client=client,
                top_k=top_k,
                mode=mode,
                threshold=threshold,
            )
        result["weather"] = None
    save_message(database_url, conversation_id, "user", question)
    save_message(
        database_url,
        conversation_id,
        "assistant",
        result["answer"],
        sources=result["sources"],
        route=result["route"],
        latency_ms=result["latency_ms"],
        cache_hit=result["cache_hit"],
    )
    if fresh:
        rename_conversation(database_url, conversation_id, question[:60])
    result["conversation_id"] = conversation_id
    result["rewritten"] = rewritten["rewritten"]
    result["rewritten_question"] = rewritten["question"]
    return result


def _weather_turn(
    question: str,
    city: str | None,
    *,
    database_url: str,
    redis_url: str | None,
) -> dict:
    start = time.monotonic()
    if not city:
        return {
            "answer": "Which city should I check the weather for?",
            "sources": [],
            "refused": False,
            "route": "weather",
            "latency_ms": _elapsed_ms(start),
            "cache_hit": False,
            "weather": None,
        }
    try:
        card = get_weather(city, database_url=database_url, redis_url=redis_url)
    except (UnknownCity, WeatherError) as exc:
        return {
            "answer": str(exc),
            "sources": [],
            "refused": False,
            "route": "weather",
            "latency_ms": _elapsed_ms(start),
            "cache_hit": False,
            "weather": None,
        }
    place = f"{card['city']}, {card['country']}".rstrip(", ")
    return {
        "answer": (
            f"{place}: {card['temp_c']}°C, {card['description'].lower()},"
            f" wind {card['wind_kph']} kph."
        ),
        "sources": [],
        "refused": False,
        "route": "weather",
        "latency_ms": _elapsed_ms(start),
        "cache_hit": card["cached"],
        "weather": card,
    }
