"""Ingestion tests: extraction, dedupe, transactions (stub client).

Live NIM is used only in the `live`-marked fixture test; everything else
runs on deterministic stub embeddings against the local Docker DB.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import psycopg
import pytest

from rag.ingest import (
    IngestError,
    extract_text,
    get_corpus_version,
    ingest_file,
    ingest_path,
)
from rag.models import EMBEDDING_DIM
from rag.nim_client import NimClient

FIXTURES = Path(__file__).parent / "fixtures"


def make_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(content)
        + content
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objs) + 1, xref_at)
    )
    return bytes(out)


class StubClient:
    """Deterministic 2048-dim embeddings; counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts, input_type="passage"):
        self.calls += 1
        vecs = []
        for t in texts:
            v = [0.0] * EMBEDDING_DIM
            v[int(hashlib.sha256(t.encode()).hexdigest(), 16) % EMBEDDING_DIM] = 1.0
            vecs.append(v)
        return vecs


@pytest.fixture
def stub():
    return StubClient()


@pytest.fixture
def cleaner(migrated_db):
    ids: list[int] = []
    yield ids
    if ids:
        with psycopg.connect(migrated_db, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM documents WHERE id = ANY(%s::bigint[])",
                (ids,),
            )


def _counts(url):
    with psycopg.connect(url) as conn:
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return docs, chunks


def test_extract_txt_and_md():
    assert extract_text(b"hello", "a.txt") == "hello"
    assert extract_text(b"# Title\nbody", "a.md") == "# Title\nbody"
    assert extract_text(b"UPPER", "A.TXT") == "UPPER"


def test_extract_pdf():
    assert extract_text(make_pdf("Hello PDF world"), "a.pdf") == (
        "Hello PDF world"
    )


def test_extract_rejects_unknown_extension():
    with pytest.raises(IngestError, match="Unsupported file type"):
        extract_text(b"data", "a.exe")


def test_extract_rejects_undecodable_text():
    with pytest.raises(IngestError, match="UTF-8"):
        extract_text(b"\xff\xfe\x00bad", "a.txt")


def test_ingest_txt_then_dedupe(migrated_db, stub, cleaner):
    before = get_corpus_version(migrated_db)
    data = b"dedupe-marker-alpha unique bytes for t003"
    first = ingest_file(
        data, "dedupe.txt", database_url=migrated_db, client=stub
    )
    cleaner.append(first["document_id"])
    assert first["deduped"] is False
    assert first["chunks_added"] == 1
    assert first["corpus_version"] == before + 1
    second = ingest_file(
        data, "dedupe.txt", database_url=migrated_db, client=stub
    )
    assert second["document_id"] == first["document_id"]
    assert second["deduped"] is True
    assert second["chunks_added"] == 0
    assert second["corpus_version"] == before + 1
    assert get_corpus_version(migrated_db) == before + 1


def test_ingest_corrupt_pdf_leaves_no_rows(migrated_db, stub):
    before = _counts(migrated_db)
    with pytest.raises(IngestError, match="PDF"):
        ingest_file(
            b"%PDF-1.4 not really a pdf {{{",
            "corrupt.pdf",
            database_url=migrated_db,
            client=stub,
        )
    assert _counts(migrated_db) == before


def test_ingest_empty_and_blank_rejected(migrated_db, stub):
    with pytest.raises(IngestError, match="empty"):
        ingest_file(b"", "e.txt", database_url=migrated_db, client=stub)
    with pytest.raises(IngestError, match="no extractable text"):
        ingest_file(b"   \n ", "b.txt", database_url=migrated_db, client=stub)


def test_ingest_records_token_counts(migrated_db, stub, cleaner):
    res = ingest_file(
        b"token count marker beta bytes here",
        "tokens.txt",
        database_url=migrated_db,
        client=stub,
    )
    cleaner.append(res["document_id"])
    with psycopg.connect(migrated_db) as conn:
        counts = conn.execute(
            "SELECT token_count FROM chunks WHERE document_id = %s",
            (res["document_id"],),
        ).fetchall()
    assert counts and all(c[0] and c[0] > 0 for c in counts)


def test_ingest_path_reads_file(tmp_path, migrated_db, stub, cleaner):
    target = tmp_path / "via-path.md"
    target.write_text("# Path ingestion marker gamma")
    res = ingest_path(target, database_url=migrated_db, client=stub)
    cleaner.append(res["document_id"])
    assert res["filename"] == "via-path.md"
    assert res["chunks_added"] == 1


def test_chunk_counts_match_splitter(migrated_db, stub, cleaner):
    from rag.chunking import split_text

    text = "alpha beta gamma delta epsilon zeta eta theta. " * 30
    res = ingest_file(
        text.encode(), "multi.txt", database_url=migrated_db, client=stub,
        chunk_size=20, chunk_overlap=5,
    )
    cleaner.append(res["document_id"])
    expected = len(split_text(text, chunk_size=20, chunk_overlap=5))
    assert res["chunks_added"] == expected > 1


@pytest.mark.live
def test_live_ingest_fixture_corpus(migrated_db):
    key = os.environ.get("NVIDIA_API_KEY")
    chat_model = os.environ.get("NIM_CHAT_MODEL")
    embed_model = os.environ.get("NIM_EMBED_MODEL")
    if not (key and chat_model and embed_model):
        pytest.skip("needs NVIDIA_API_KEY + NIM_CHAT_MODEL + NIM_EMBED_MODEL")
    client = NimClient(chat_model=chat_model, embed_model=embed_model)
    ids = []
    try:
        for name in ("rivers.txt", "curie.txt", "bridges.txt"):
            data = (FIXTURES / name).read_bytes()
            res = ingest_file(
                data, name, database_url=migrated_db, client=client
            )
            ids.append(res["document_id"])
            assert res["chunks_added"] >= 1
            print(f"\n{name}: {res['chunks_added']} chunks")
    finally:
        if ids:
            with psycopg.connect(migrated_db, autocommit=True) as conn:
                conn.execute(
                    "DELETE FROM documents WHERE id = ANY(%s::bigint[])",
                    (ids,),
                )
