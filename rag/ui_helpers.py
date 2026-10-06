"""Pure UI helpers: formatting, filenames, friendly error mapping.

The UI must never show a traceback: every backend error maps to a short
human message via `friendly_error()`. Unknown errors map to a generic
note without leaking internals.
"""

from __future__ import annotations

import re

import psycopg
import streamlit as st

from rag.chat import HistoryError
from rag.config import load_config
from rag.ingest import IngestError
from rag.nim_client import NimAuthError, NimClient, NimError
from rag.prompts import PromptError
from rag.tools_weather import UnknownCity, WeatherError


def format_citation(source: dict) -> str:
    """Format a source as [doc name #chunk]."""
    return f"[{source['doc_name']} #{source['chunk_index']}]"


def badge_text(
    route: str | None, latency_ms: int | None, cache_hit: bool
) -> str:
    """One-line route/latency/cache badge ('' when empty)."""
    parts = []
    if route:
        parts.append(route)
    if latency_ms is not None:
        parts.append(f"{latency_ms} ms")
    if cache_hit:
        parts.append("cache hit")
    return " · ".join(parts)


def export_filename(title: str) -> str:
    """Slugify a title into a .md filename."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")
    return f"{slug or 'chat'}.md"


REQUIRED_SETTINGS = (
    ("DATABASE_URL", "database_url"),
    ("NVIDIA_API_KEY", "nvidia_api_key"),
    ("NIM_CHAT_MODEL", "nim_chat_model"),
    ("NIM_EMBED_MODEL", "nim_embed_model"),
)


def load_or_error():
    """Load config, or return (None, message) when settings are missing."""
    cfg = load_config()
    missing = [name for name, attr in REQUIRED_SETTINGS if not getattr(cfg, attr)]
    if missing:
        return None, (
            f"Missing configuration: {', '.join(missing)}. "
            "Set them in your environment (see .env.example) and rerun."
        )
    return cfg, None


@st.cache_resource
def get_nim_client(api_key, base_url, chat_model, embed_model):
    """Layer-5 cached NIM client shared by all pages."""
    return NimClient(
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        embed_model=embed_model,
    )


def friendly_error(exc: BaseException) -> str:
    """Map any backend error to a short human-safe message."""
    if isinstance(exc, NimAuthError):
        return (
            "The model service rejected the credentials. "
            "Check NVIDIA_API_KEY and try again."
        )
    if isinstance(exc, NimError):
        return f"The model service had trouble: {exc} Please try again."
    if isinstance(exc, (IngestError, UnknownCity, WeatherError)):
        return str(exc)
    if isinstance(exc, HistoryError):
        return f"History problem: {exc}"
    if isinstance(exc, PromptError):
        return f"Prompt configuration problem: {exc}"
    if isinstance(exc, psycopg.OperationalError):
        return (
            "The database is unreachable. "
            "Check DATABASE_URL and try again."
        )
    if isinstance(exc, psycopg.Error):
        return "The database returned an error. Please try again."
    return "Something unexpected happened. Please try again."
