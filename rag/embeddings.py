"""Batched document embedding over :class:`rag.nim_client.NimClient`."""

from __future__ import annotations

from collections.abc import Sequence

from rag.nim_client import NimClient


def batch_embed(
    client: NimClient,
    texts: Sequence[str],
    *,
    input_type: str = "passage",
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed *texts* in batches of *batch_size* (empty input -> ``[]``)."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    items = list(texts)
    if not items:
        return []
    out: list[list[float]] = []
    for start in range(0, len(items), batch_size):
        out.extend(
            client.embed(items[start : start + batch_size], input_type)
        )
    return out
