"""Layer-4 tool cache: short-TTL JSON payloads.

Redis when `REDIS_URL` is set, else the `tool_cache` table. Standalone
module (no `chat` dependency) so tools can use it without import cycles.
"""

from __future__ import annotations

import json
import logging

import psycopg
import redis

logger = logging.getLogger(__name__)

REDIS_TOOL_PREFIX = "rag:tool:"
DEFAULT_TOOL_TTL_SECONDS = 600


def _redis_client(redis_url: str) -> redis.Redis:
    return redis.Redis.from_url(redis_url, socket_timeout=5)


def tool_get(
    database_url: str, key: str, *, redis_url: str | None = None
):
    """Read a cached payload; None on miss or expiry."""
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
    """Write a cached payload with TTL."""
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


def clear_tool_cache(
    database_url: str, *, redis_url: str | None = None
) -> dict[str, int]:
    """Empty the tool cache; returns cleared counts."""
    with psycopg.connect(database_url, autocommit=True) as conn:
        rows = conn.execute("DELETE FROM tool_cache RETURNING key").fetchall()
    counts = {"tool_cache": len(rows), "redis_keys": 0}
    if redis_url:
        client = _redis_client(redis_url)
        names = list(client.scan_iter(REDIS_TOOL_PREFIX + "*", count=1000))
        if names:
            counts["redis_keys"] = client.delete(*names)
    return counts
