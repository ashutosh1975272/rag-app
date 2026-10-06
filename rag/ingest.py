"""Ingestion: extract → chunk → embed → store, with SHA-256 dedupe.

One new document = one transaction (document + chunks + corpus bump), so a
failure never leaves partial rows. Re-ingesting identical bytes returns the
existing document id without touching the corpus version.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import psycopg
from pypdf import PdfReader

from rag.chunking import count_tokens, split_text
from rag.embeddings import batch_embed
from rag.vectorstore import to_vector_literal

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
PDF_EXTENSION = ".pdf"


class IngestError(ValueError):
    """Friendly ingestion failure (bad file, unsupported type, no text)."""


def extract_text(data: bytes, filename: str) -> str:
    """Extract text from PDF/TXT/MD bytes by filename extension."""
    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise IngestError(
                f"{filename}: could not decode as UTF-8 text."
            ) from None
    if suffix == PDF_EXTENSION:
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # noqa: BLE001 - any parse failure is a bad PDF
            raise IngestError(
                f"{filename}: could not parse as PDF ({type(exc).__name__})."
            ) from None
    raise IngestError(
        f"{filename}: Unsupported file type {suffix or '(none)'}; "
        "use .pdf, .txt, or .md."
    )


def get_corpus_version(database_url: str) -> int:
    """Current corpus version from app_meta."""
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            "SELECT value FROM app_meta WHERE key = 'corpus_version'"
        ).fetchone()
    if row is None:
        raise IngestError("app_meta.corpus_version missing: run migrations.")
    return int(row[0])


def _fetch_existing(conn: psycopg.Connection, sha: str):
    return conn.execute(
        "SELECT id, filename FROM documents WHERE sha256 = %s", (sha,)
    ).fetchone()


def ingest_file(
    data: bytes,
    filename: str,
    *,
    database_url: str,
    client,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    embed_model: str | None = None,
) -> dict:
    """Ingest one file; returns document summary (see module docstring)."""
    if not data:
        raise IngestError(f"{filename}: file is empty.")
    text = extract_text(data, filename)
    if not text.strip():
        raise IngestError(
            f"{filename}: no extractable text "
            "(scanned-image PDFs are not supported)."
        )
    sha = hashlib.sha256(data).hexdigest()
    chunks = split_text(text, chunk_size, chunk_overlap)
    vectors = batch_embed(
        client,
        chunks,
        input_type="passage",
        database_url=database_url,
        embed_model=embed_model,
    )
    counts = [count_tokens(c) for c in chunks]

    with psycopg.connect(database_url) as conn:
        with conn.transaction():
            row = conn.execute(
                "INSERT INTO documents (filename, sha256, size_bytes)"
                " VALUES (%s, %s, %s)"
                " ON CONFLICT (sha256) DO NOTHING RETURNING id",
                (filename, sha, len(data)),
            ).fetchone()
            if row is None:
                existing = _fetch_existing(conn, sha)
                assert existing is not None
                version = conn.execute(
                    "SELECT value FROM app_meta WHERE key = 'corpus_version'"
                ).fetchone()
                assert version is not None
                return {
                    "document_id": existing[0],
                    "filename": existing[1],
                    "sha256": sha,
                    "chunks_added": 0,
                    "deduped": True,
                    "corpus_version": int(version[0]),
                }
            doc_id = row[0]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks"
                    " (document_id, ord, text, token_count, embedding)"
                    " VALUES (%s, %s, %s, %s, %s::vector)",
                    [
                        (doc_id, i, chunk, counts[i], to_vector_literal(vec))
                        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
                    ],
                )
            version = conn.execute(
                "UPDATE app_meta SET value = ((value::bigint) + 1)::text"
                " WHERE key = 'corpus_version' RETURNING value"
            ).fetchone()
            assert version is not None
        return {
            "document_id": doc_id,
            "filename": filename,
            "sha256": sha,
            "chunks_added": len(chunks),
            "deduped": False,
            "corpus_version": int(version[0]),
        }


def ingest_path(
    path: str | Path,
    *,
    database_url: str,
    client,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    embed_model: str | None = None,
) -> dict:
    """Read *path* from disk and ingest it under its own filename."""
    target = Path(path)
    return ingest_file(
        target.read_bytes(),
        target.name,
        database_url=database_url,
        client=client,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
    )


__all__ = [
    "IngestError",
    "extract_text",
    "get_corpus_version",
    "ingest_file",
    "ingest_path",
]
