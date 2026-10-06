"""Caching layers 1–4 (layer 5, `st.cache_resource`, lives in the UI).

1. Embedding cache: same text+model is never embedded twice.
2. Exact answer cache: keyed by normalized question + corpus + model +
   prompt versions.
3. Semantic answer cache: cosine ≥ threshold on the stored question
   embedding, same corpus and prompt stamp.
4. Tool cache: short-TTL JSON payloads (Redis when `REDIS_URL` is set,
   else the `tool_cache` table).

Every hit/miss is logged at INFO with its layer name.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

import psycopg
import redis
from pgvector.psycopg import register_vector

from rag.chat import DEFAULT_THRESHOLD, REFUSAL, ask
from rag.ingest import get_corpus_version
from rag.prompts import get_active, seed_defaults
from rag.vectorstore import to_vector_literal

logger = logging.getLogger(__name__)

REDIS_TOOL_PREFIX = "rag:tool:"
DEFAULT_TOOL_TTL_SECONDS = 600


def norm_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_key(
    question: str,
    corpus_version: int,
    model: str,
    prompt_versions: dict,
) -> str:
    stamp = f"sys{prompt_versions.get('system')}:ans{prompt_versions.get('answer')}"
    return text_sha(f"{norm_question(question)}|{corpus_version}|{model}|{stamp}")


def prompt_stamp(
    chat_model: str, prompt_versions: dict
) -> str:
    return (
        f"{chat_model}|sys{prompt_versions.get('system')}"
        f"|ans{prompt_versions.get('answer')}"
    )


def cached_embed_texts(
    client,
    texts: list[str],
    *,
    database_url: str,
    model: str,
    input_type: str = "passage",
    batch_size: int = 32,
) -> list[list[float]]:
    """Layer 1: embed only cache misses, then merge in order."""
    if not texts:
        return []
    shas = [text_sha(t) for t in texts]
    by_sha: dict[str, list[float]] = {}
    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        rows = conn.execute(
            "SELECT text_sha256, embedding, model FROM embedding_cache"
            " WHERE text_sha256 = ANY(%s)",
            (shas,),
        ).fetchall()
        for sha, vec, stored_model in rows:
            if stored_model == model:
                by_sha[sha] = vec.to_list()
    logger.info(
        "cache.embed %d hits, %d misses (model %s)",
        len(by_sha),
        len(set(shas) - set(by_sha)),
        model,
    )
    missing = [t for t, s in zip(texts, shas) if s not in by_sha]
    fresh: dict[str, list[float]] = {}
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        for text, vec in zip(batch, client.embed(batch, input_type)):
            fresh[text_sha(text)] = vec
    if fresh:
        with (
            psycopg.connect(database_url, autocommit=True) as conn,
            conn.cursor() as cur,
        ):
            cur.executemany(
                "INSERT INTO embedding_cache"
                " (text_sha256, model, embedding) VALUES (%s, %s, %s::vector)"
                " ON CONFLICT (text_sha256) DO UPDATE SET model = %s,"
                " embedding = %s::vector",
                [
                    (
                        sha,
                        model,
                        to_vector_literal(vec),
                        model,
                        to_vector_literal(vec),
                    )
                    for sha, vec in fresh.items()
                ],
            )
    by_sha.update(fresh)
    return [by_sha[s] for s in shas]


def cached_ask(
    question: str,
    *,
    database_url: str,
    client,
    chat_model: str,
    embed_model: str,
    top_k: int = 6,
    mode: str = "hybrid",
    threshold: float = DEFAULT_THRESHOLD,
    semantic_threshold: float = 0.95,
) -> dict:
    """Layers 2+3 around `ask()`; adds `cache_hit`/`cache_layer`."""
    seed_defaults(database_url)
    corpus = get_corpus_version(database_url)
    sys_v, _ = get_active(database_url, "system")
    ans_v, _ = get_active(database_url, "answer")
    versions = {"system": sys_v, "answer": ans_v}
    key = exact_key(question, corpus, chat_model, versions)
    stamp = prompt_stamp(chat_model, versions)

    with psycopg.connect(database_url, autocommit=True) as conn:
        row = conn.execute(
            "SELECT answer, sources FROM answer_cache"
            " WHERE key = %s AND corpus_version = %s",
            (key, corpus),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE answer_cache SET hits = hits + 1 WHERE key = %s",
                (key,),
            )
            logger.info("cache.answer exact hit (corpus %s)", corpus)
            return {
                "answer": row[0],
                "sources": row[1] or [],
                "refused": row[0] == REFUSAL,
                "prompt_versions": versions,
                "route": "rag",
                "latency_ms": 0,
                "cache_hit": True,
                "cache_layer": "exact",
            }

    query_vec = cached_embed_texts(
        client,
        [question],
        database_url=database_url,
        model=embed_model,
        input_type="query",
    )[0]
    with psycopg.connect(database_url, autocommit=True) as conn:
        register_vector(conn)
        row = conn.execute(
            "SELECT key, answer, sources,"
            " 1 - (question_embedding <=> %s::vector) AS sim"
            " FROM answer_cache WHERE corpus_version = %s"
            " AND prompt_stamp = %s AND question_embedding IS NOT NULL"
            " ORDER BY question_embedding <=> %s::vector LIMIT 1",
            (
                to_vector_literal(query_vec),
                corpus,
                stamp,
                to_vector_literal(query_vec),
            ),
        ).fetchone()
        if row is not None and float(row[3]) >= semantic_threshold:
            conn.execute(
                "UPDATE answer_cache SET hits = hits + 1 WHERE key = %s",
                (row[0],),
            )
            logger.info(
                "cache.answer semantic hit sim=%.3f (corpus %s)",
                float(row[3]),
                corpus,
            )
            return {
                "answer": row[1],
                "sources": row[2] or [],
                "refused": row[1] == REFUSAL,
                "prompt_versions": versions,
                "route": "rag",
                "latency_ms": 0,
                "cache_hit": True,
                "cache_layer": "semantic",
            }

    logger.info("cache.answer miss (corpus %s)", corpus)
    result = ask(
        question,
        database_url=database_url,
        client=client,
        top_k=top_k,
        mode=mode,
        threshold=threshold,
    )
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO answer_cache"
            " (key, question_norm, question_embedding, answer, sources,"
            " corpus_version, prompt_stamp)"
            " VALUES (%s, %s, %s::vector, %s, %s::jsonb, %s, %s)"
            " ON CONFLICT (key) DO NOTHING",
            (
                key,
                norm_question(question),
                to_vector_literal(query_vec),
                result["answer"],
                json.dumps(result["sources"]),
                corpus,
                stamp,
            ),
        )
    result["cache_hit"] = False
    result["cache_layer"] = None
    return result


def _redis_client(redis_url: str) -> redis.Redis:
    return redis.Redis.from_url(redis_url, socket_timeout=5)


def tool_get(
    database_url: str, key: str, *, redis_url: str | None = None
):
    """Layer 4 read; None on miss or expiry."""
    if redis_url:
        raw = _redis_client(redis_url).get(REDIS_TOOL_PREFIX + key)
        hit = raw is not None
        logger.info("cache.tool redis %s: %s", "hit" if hit else "miss", key)
        return json.loads(raw) if hit else None
    with psycopg.connect(database_url, autocommit=True) as conn:
        row = conn.execute(
            "SELECT payload, expires_at > now() AS fresh FROM tool_cache"
            " WHERE key = %s",
            (key,),
        ).fetchone()
        if row is None:
            logger.info("cache.tool postgres miss: %s", key)
            return None
        if not row[1]:
            conn.execute("DELETE FROM tool_cache WHERE key = %s", (key,))
            logger.info("cache.tool postgres expired: %s", key)
            return None
        logger.info("cache.tool postgres hit: %s", key)
        return row[0]


def tool_set(
    database_url: str,
    key: str,
    payload: dict,
    *,
    ttl_seconds: int = DEFAULT_TOOL_TTL_SECONDS,
    redis_url: str | None = None,
) -> None:
    """Layer 4 write with TTL."""
    if redis_url:
        client = _redis_client(redis_url)
        name = REDIS_TOOL_PREFIX + key
        if ttl_seconds <= 0:
            client.delete(name)
        else:
            client.setex(name, ttl_seconds, json.dumps(payload))
        return
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO tool_cache (key, payload, expires_at)"
            " VALUES (%s, %s::jsonb, now() + %s * INTERVAL '1 second')"
            " ON CONFLICT (key) DO UPDATE SET payload = %s::jsonb,"
            " expires_at = now() + %s * INTERVAL '1 second'",
            (
                key,
                json.dumps(payload),
                ttl_seconds,
                json.dumps(payload),
                ttl_seconds,
            ),
        )


def clear_caches(
    database_url: str, *, redis_url: str | None = None
) -> dict[str, int]:
    """Empty layers 1–4; returns per-store cleared counts."""
    key_columns = {
        "embedding_cache": "text_sha256",
        "answer_cache": "key",
        "tool_cache": "key",
    }
    with psycopg.connect(database_url, autocommit=True) as conn:
        counts = {}
        for table, key_column in key_columns.items():
            # Table/column names come from the hardcoded dict above.
            rows = conn.execute(
                f"DELETE FROM {table} RETURNING {key_column}"
            ).fetchall()
            counts[table] = len(rows)
    redis_keys = 0
    if redis_url:
        client = _redis_client(redis_url)
        names = list(client.scan_iter(REDIS_TOOL_PREFIX + "*", count=1000))
        if names:
            redis_keys = client.delete(*names)
    counts["redis_keys"] = redis_keys
    logger.info("cache.clear %s", counts)
    return counts
