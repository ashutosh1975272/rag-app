"""Chunking tests: exact pinned splits + edge cases (no DB, no keys)."""

import pytest

from rag.chunking import make_splitter, split_text

SENTENCES = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump!"
)


def test_exact_chunks_with_overlap():
    assert split_text(SENTENCES, chunk_size=20, chunk_overlap=5) == [
        (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs."
        ),
        "dozen liquor jugs. How vexingly quick daft zebras jump!",
    ]


def test_short_text_is_single_chunk():
    assert split_text("short", chunk_size=20, chunk_overlap=5) == ["short"]


def test_blank_text_is_empty():
    assert split_text("   ", chunk_size=20, chunk_overlap=5) == []
    assert split_text("", chunk_size=20, chunk_overlap=5) == []


def test_overlap_tail_overlaps_next_head():
    chunks = split_text(SENTENCES, chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 2
    assert chunks[1].startswith("dozen liquor jugs.")
    assert chunks[0].endswith("dozen liquor jugs.")


def test_invalid_sizes_rejected():
    with pytest.raises(ValueError):
        make_splitter(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        make_splitter(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        make_splitter(chunk_size=10, chunk_overlap=-1)
