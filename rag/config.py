"""App configuration: environment variables with defaults.

Secrets are read from the environment only; this module never prints values.
Use require() at the point a setting is actually needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


class ConfigError(ValueError):
    """Raised when a setting is missing or malformed (name only, no value)."""


@dataclass(frozen=True)
class Config:
    database_url: str | None = None
    redis_url: str | None = None
    nvidia_api_key: str | None = None
    github_token: str | None = None
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_chat_model: str | None = None
    nim_embed_model: str | None = None
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 6
    semantic_threshold: float = 0.95


def _get_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"Invalid integer for {name}") from None


def _get_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"Invalid float for {name}") from None


def load_config(env: Mapping[str, str] | None = None) -> Config:
    src: Mapping[str, str] = os.environ if env is None else env
    return Config(
        database_url=src.get("DATABASE_URL") or None,
        redis_url=src.get("REDIS_URL") or None,
        nvidia_api_key=src.get("NVIDIA_API_KEY") or None,
        github_token=src.get("GITHUB_TOKEN") or None,
        nim_base_url=src.get("NIM_BASE_URL") or Config.nim_base_url,
        nim_chat_model=src.get("NIM_CHAT_MODEL") or None,
        nim_embed_model=src.get("NIM_EMBED_MODEL") or None,
        chunk_size=_get_int(src, "CHUNK_SIZE", 800),
        chunk_overlap=_get_int(src, "CHUNK_OVERLAP", 100),
        top_k=_get_int(src, "TOP_K", 6),
        semantic_threshold=_get_float(src, "SEMANTIC_THRESHOLD", 0.95),
    )


def require(value: str | None, name: str) -> str:
    if not value:
        raise ConfigError(f"Missing required setting: {name}")
    return value
