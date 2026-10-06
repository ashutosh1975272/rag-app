"""Batched document embedding over :class:`rag.nim_client.NimClient`.

Pass ``database_url`` + ``embed_model`` to route through the layer-1
embedding cache (same text+model embedded once); otherwise embeds directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.nim_client import NimClient


def batch_embed(
    client: NimClient,
    texts: Sequence[str],
    *,
    input_type: str = "passage",
    batch_size: int = 32,
    database_url: str | None = None,
    embed_model: str | None = None,
) -> list[list[float]]:
    """Embed *texts* in batches of *batch_size* (empty input -> ``[]``)."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    items = list(texts)
    if not items:
        return []
    if database_url is not None and embed_model is not None:
        from rag.cache import cached_embed_texts

        return cached_embed_texts(
            client,
            items,
            database_url=database_url,
            model=embed_model,
            input_type=input_type,
            batch_size=batch_size,
        )
    out: list[list[float]] = []
    for start in range(0, len(items), batch_size):
        out.extend(client.embed(items[start : start + batch_size], input_type))
    return out
