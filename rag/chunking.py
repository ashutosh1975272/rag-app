"""Text chunking via LangChain's tiktoken-backed recursive splitter.

Sizes are tokens (cl100k_base), matching the ``CHUNK_SIZE``/``CHUNK_OVERLAP``
settings. No silent fallback: if tiktoken cannot load, construction fails
loudly instead of silently changing the unit to characters.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_splitter(
    chunk_size: int = 800, chunk_overlap: int = 100
) -> RecursiveCharacterTextSplitter:
    """Build a recursive splitter with tiktoken lengths."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must satisfy 0 <= overlap < size, got "
            f"{chunk_overlap} / {chunk_size}"
        )
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def split_text(
    text: str, chunk_size: int = 800, chunk_overlap: int = 100
) -> list[str]:
    """Split *text* into overlapping chunks (empty input -> ``[]``)."""
    if not text or not text.strip():
        return []
    return make_splitter(chunk_size, chunk_overlap).split_text(text)
