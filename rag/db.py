"""Postgres connections: psycopg3 pool + SQLAlchemy engine (psycopg driver).

Only reads connection strings; never prints them.
"""

from __future__ import annotations

from contextlib import contextmanager
from urllib.parse import urlparse

from psycopg_pool import ConnectionPool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local_database_url(url: str) -> bool:
    """True only for localhost Postgres URLs (fail-closed guard for tests)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in LOCAL_HOSTS


def sqlalchemy_url(database_url: str) -> str:
    """Map a ``postgresql://`` URL to the SQLAlchemy psycopg-dialect URL."""
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url[len("postgresql://") :]
    return database_url


@contextmanager
def get_pool(database_url: str, min_size: int = 1, max_size: int = 5):
    """Yield a psycopg3 ConnectionPool, closed on exit."""
    pool = ConnectionPool(
        database_url, min_size=min_size, max_size=max_size, open=True
    )
    try:
        yield pool
    finally:
        pool.close()


def health(pool: ConnectionPool) -> bool:
    """True when a pooled connection runs SELECT 1."""
    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1")
    except Exception:  # noqa: BLE001 - any failure means unhealthy
        return False
    return True


@contextmanager
def get_engine(database_url: str):
    """Yield a SQLAlchemy Engine (QueuePool, pre-ping), disposed on exit."""
    engine = create_engine(sqlalchemy_url(database_url), pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()


@contextmanager
def get_session(engine: Engine):
    """Yield a SQLAlchemy Session bound to *engine*, closed on exit."""
    with Session(engine) as session:
        yield session


def health_engine(engine: Engine) -> bool:
    """True when the engine runs SELECT 1."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any failure means unhealthy
        return False
    return True
