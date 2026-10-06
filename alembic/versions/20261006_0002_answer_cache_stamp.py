"""answer_cache prompt stamp for semantic-hit invalidation (T-007).

Revision ID: 20261006_0002
Revises: 20261006_0001
Create Date: 2026-10-06

Semantic cache hits must match on chat model + prompt versions, not just
corpus; otherwise a prompt edit leaves stale semantic hits. Pre-existing
rows keep NULL stamps and stay exact-matchable only.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20261006_0002"
down_revision: str | None = "20261006_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("answer_cache", sa.Column("prompt_stamp", sa.Text))


def downgrade() -> None:
    op.drop_column("answer_cache", "prompt_stamp")
