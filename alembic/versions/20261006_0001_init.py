"""init: full app schema (T-001).

Revision ID: 20261006_0001
Revises:
Create Date: 2026-10-06

Embedding width vector(2048) is the measured NIM dimension
(MEMORY.md 2026-10-05), hardcoded so history stays immutable.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20261006_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column(
            "id", sa.BigInteger, sa.Identity(always=False), primary_key=True
        ),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("sha256", sa.Text, nullable=False, unique=True),
        sa.Column("size_bytes", sa.BigInteger),
        sa.Column(
            "status",
            sa.Text,
            nullable=False,
            server_default=sa.text("'ready'"),
        ),
        sa.Column("corpus_version", sa.BigInteger),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "chunks",
        sa.Column(
            "id", sa.BigInteger, sa.Identity(always=False), primary_key=True
        ),
        sa.Column(
            "document_id",
            sa.BigInteger,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer),
        sa.Column("embedding", Vector(2048)),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', text)", persisted=True),
        ),
    )
    # No vector index: pgvector caps HNSW/IVFFlat at 2000 dims, but NIM
    # embeddings are 2048-dim (measured). Exact cosine search (`<=>`) needs
    # no index and is fine at this corpus size; revisit halfvec+HNSW if it
    # grows. (Proven live: "column cannot have more than 2000 dimensions
    # for hnsw".)
    op.create_index(
        "chunks_tsv_gin", "chunks", ["tsv"], postgresql_using="gin"
    )
    op.create_index("chunks_document_id", "chunks", ["document_id"])

    op.create_table(
        "conversations",
        sa.Column(
            "id", sa.BigInteger, sa.Identity(always=False), primary_key=True
        ),
        sa.Column(
            "title",
            sa.Text,
            nullable=False,
            server_default=sa.text("'New chat'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "messages",
        sa.Column(
            "id", sa.BigInteger, sa.Identity(always=False), primary_key=True
        ),
        sa.Column(
            "conversation_id",
            sa.BigInteger,
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "sources",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("route", sa.Text),
        sa.Column("latency_ms", sa.Integer),
        sa.Column(
            "cache_hit", sa.Boolean, nullable=False, server_default=sa.false()
        ),
        sa.Column("feedback", sa.SmallInteger),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("messages_conversation_id", "messages", ["conversation_id"])

    op.create_table(
        "embedding_cache",
        sa.Column("text_sha256", sa.Text, primary_key=True),
        sa.Column("model", sa.Text, nullable=False),
        sa.Column("embedding", Vector(2048), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "answer_cache",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("question_norm", sa.Text, nullable=False),
        sa.Column("question_embedding", Vector(2048)),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column(
            "sources",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("corpus_version", sa.BigInteger, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "hits", sa.Integer, nullable=False, server_default=sa.text("0")
        ),
    )

    op.create_table(
        "tool_cache",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "prompt_templates",
        sa.Column("name", sa.Text, primary_key=True),
        sa.Column("version", sa.Integer, primary_key=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "active", sa.Boolean, nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "prompt_templates_one_active",
        "prompt_templates",
        ["name"],
        unique=True,
        postgresql_where=sa.text("active"),
    )

    op.create_table(
        "app_meta",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("value", sa.Text, nullable=False),
    )
    op.execute(
        "INSERT INTO app_meta (key, value) VALUES ('corpus_version', '1')"
        " ON CONFLICT (key) DO NOTHING"
    )


def downgrade() -> None:
    op.drop_table("app_meta")
    op.drop_index("prompt_templates_one_active", table_name="prompt_templates")
    op.drop_table("prompt_templates")
    op.drop_table("tool_cache")
    op.drop_table("answer_cache")
    op.drop_table("embedding_cache")
    op.drop_index("messages_conversation_id", table_name="messages")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_index("chunks_document_id", table_name="chunks")
    op.drop_index("chunks_tsv_gin", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
