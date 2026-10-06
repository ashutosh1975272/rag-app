"""NIM discovery probe: verify configured models, measure dimensions.

Re-run whenever the key is replaced (MEMORY.md). Prints model ids and
dimensions only — never the API key. Exits non-zero when a configured id
is missing from the list or a measured dim differs from ``EMBEDDING_DIM``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

from rag.config import require
from rag.models import EMBEDDING_DIM
from rag.nim_client import NimClient

PROBE_SENTENCE = "dimension probe sentence"


def fetch_models(base_url: str, api_key: str, timeout: int = 60) -> list[str]:
    """List model ids from the NIM models endpoint."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + api_key},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    return [m["id"] for m in payload.get("data", [])]


def check_configured(
    models: list[str], chat_id: str, embed_id: str
) -> dict[str, object]:
    """Report whether the configured ids are listed."""
    return {
        "count": len(models),
        "chat_ok": chat_id in models,
        "embed_ok": embed_id in models,
    }


def suggest_alternatives(models: list[str]) -> dict[str, list[str]]:
    """Heuristic embed/chat candidates from a model list."""
    embed = sorted(m for m in models if "embed" in m.lower())
    chat = sorted(
        m
        for m in models
        if "instruct" in m.lower()
        or "chat" in m.lower()
        or "nemotron" in m.lower()
    )
    return {"embed": embed, "chat": chat}


def main() -> int:
    """Run the discovery probe; exit 0 on success."""
    base_url = os.environ.get("NIM_BASE_URL") or (
        "https://integrate.api.nvidia.com/v1"
    )
    api_key = require(os.environ.get("NVIDIA_API_KEY"), "NVIDIA_API_KEY")
    chat_id = require(os.environ.get("NIM_CHAT_MODEL"), "NIM_CHAT_MODEL")
    embed_id = require(os.environ.get("NIM_EMBED_MODEL"), "NIM_EMBED_MODEL")

    models = fetch_models(base_url, api_key)
    status = check_configured(models, chat_id, embed_id)
    print(f"models listed: {status['count']}")
    print(f"chat {chat_id}: {'OK' if status['chat_ok'] else 'MISSING'}")
    print(f"embed {embed_id}: {'OK' if status['embed_ok'] else 'MISSING'}")
    if not (status["chat_ok"] and status["embed_ok"]):
        sug = suggest_alternatives(models)
        print("embed candidates:", ", ".join(sug["embed"][:10]) or "none")
        print("chat candidates:", ", ".join(sug["chat"][:10]) or "none")
        return 1

    client = NimClient(
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_id,
        embed_model=embed_id,
    )
    passage = client.embed([PROBE_SENTENCE], "passage")[0]
    query = client.embed([PROBE_SENTENCE], "query")[0]
    print(f"passage dim: {len(passage)}")
    print(f"query dim: {len(query)}")
    print(f"query==passage: {list(passage) == list(query)}")
    if len(passage) != EMBEDDING_DIM or len(query) != EMBEDDING_DIM:
        print(f"dimension mismatch: expected {EMBEDDING_DIM}")
        return 1
    print("probe: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
