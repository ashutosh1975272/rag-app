"""Alembic migration runner.

Applies ``alembic/versions/*.py`` up to head. Re-running applies nothing.
Returns the applied revision ids in upgrade order (``[]`` when already
at head). Equivalent CLI: ``DATABASE_URL=... alembic upgrade head``.
"""

from __future__ import annotations

import pathlib

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from alembic import command
from rag.db import sqlalchemy_url

ALEMBIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "alembic"


def _config(database_url: str, alembic_dir: pathlib.Path) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(alembic_dir))
    cfg.set_main_option("sqlalchemy.url", sqlalchemy_url(database_url))
    return cfg


def _current_revision(database_url: str, script: ScriptDirectory) -> str | None:
    engine = create_engine(sqlalchemy_url(database_url))
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn, opts={"script": script})
            return ctx.get_current_revision()
    finally:
        engine.dispose()


def migrate(
    database_url: str, alembic_dir: str | None = None
) -> list[str]:
    """Upgrade to head; return applied revisions ([] if current)."""
    directory = pathlib.Path(alembic_dir) if alembic_dir else ALEMBIC_DIR
    cfg = _config(database_url, directory)
    script = ScriptDirectory.from_config(cfg)
    before = _current_revision(database_url, script)
    command.upgrade(cfg, "head")
    after = _current_revision(database_url, script)
    if before == after:
        return []
    applied = [
        rev.revision
        for rev in script.walk_revisions(base=before or "base", head=after)
    ]
    return list(reversed(applied))
