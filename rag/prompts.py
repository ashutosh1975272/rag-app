"""Versioned prompt templates in `prompt_templates`.

Versions are insert-only (never updated in place); exactly one version per
name is active. `seed_defaults()` is idempotent: safe to call on every ask.
"""

from __future__ import annotations

import psycopg

SYSTEM_V1 = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context excerpts. If the context does not contain the answer, "
    "reply exactly: I don't know based on the uploaded documents."
)

ANSWER_V1 = (
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only "
    "the context above. Do not invent facts or sources."
)

REWRITE_V1 = (
    "Rewrite the follow-up question as a standalone question that keeps "
    "its full meaning without the conversation history. Reply with only "
    "the rewritten question.\n\nConversation:\n{history}\n\nFollow-up: "
    "{question}"
)

ROUTER_V1 = (
    "Classify the user message as exactly one of: rag, weather, chitchat.\n"
    "- weather: asks about weather, temperature, or forecast for a place.\n"
    "- chitchat: greeting, thanks, farewell, or small talk.\n"
    "- rag: anything else (questions about documents or facts).\n\n"
    "Message: {message}\n\nReply with only the label."
)

DEFAULTS = {
    "system": SYSTEM_V1,
    "answer": ANSWER_V1,
    "query-rewrite": REWRITE_V1,
    "router": ROUTER_V1,
}


class PromptError(ValueError):
    """Prompt template missing or misconfigured."""


def seed_defaults(database_url: str) -> None:
    """Insert v1 for missing names; activate where none is active."""
    with psycopg.connect(database_url, autocommit=True) as conn:
        for name, content in DEFAULTS.items():
            conn.execute(
                "INSERT INTO prompt_templates (name, version, content)"
                " VALUES (%s, 1, %s) ON CONFLICT DO NOTHING",
                (name, content),
            )
            conn.execute(
                "UPDATE prompt_templates SET active = TRUE"
                " WHERE name = %s AND version = 1"
                " AND NOT EXISTS (SELECT 1 FROM prompt_templates p"
                " WHERE p.name = %s AND p.active)",
                (name, name),
            )


def get_active(database_url: str, name: str) -> tuple[int, str]:
    """Return (version, content) of the active template for *name*."""
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            "SELECT version, content FROM prompt_templates"
            " WHERE name = %s AND active",
            (name,),
        ).fetchone()
    if row is None:
        raise PromptError(f"No active prompt template: {name!r}")
    return row[0], row[1]


def add_version(
    database_url: str, name: str, content: str, *, activate: bool = True
) -> int:
    """Insert a new version; optionally make it the active one."""
    with psycopg.connect(database_url) as conn:
        with conn.transaction():
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM prompt_templates"
                " WHERE name = %s",
                (name,),
            ).fetchone()
            assert row is not None
            version = row[0] + 1
            conn.execute(
                "INSERT INTO prompt_templates (name, version, content)"
                " VALUES (%s, %s, %s)",
                (name, version, content),
            )
            if activate:
                conn.execute(
                    "UPDATE prompt_templates SET active = (version = %s)"
                    " WHERE name = %s",
                    (version, name),
                )
    return version
