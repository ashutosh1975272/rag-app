"""UI smoke tests: AppTest flows over the real test database.

Backend model calls are stubbed (no network); database access is real.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from rag import chat as chat_api
from rag.nim_client import NimAuthError

ROOT = Path(__file__).parent.parent


@pytest.fixture
def ui_env(migrated_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", migrated_db)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("NIM_CHAT_MODEL", "test-chat")
    monkeypatch.setenv("NIM_EMBED_MODEL", "test-embed")
    monkeypatch.delenv("REDIS_URL", raising=False)
    st.cache_resource.clear()
    with psycopg.connect(migrated_db) as conn:
        before = {
            r[0]
            for r in conn.execute("SELECT id FROM conversations").fetchall()
        }
    yield migrated_db
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        after = {
            r[0]
            for r in conn.execute("SELECT id FROM conversations").fetchall()
        }
        for cid in after - before:
            conn.execute("DELETE FROM conversations WHERE id = %s", (cid,))
    st.cache_resource.clear()


def _fake_answer(prompt, **kwargs):
    url = os.environ["DATABASE_URL"]
    cid = kwargs.get("conversation_id") or chat_api.create_conversation(url)[
        "id"
    ]
    answer = f"CANNED reply to: {prompt}"
    sources = [{"doc_name": "d.txt", "chunk_index": 0, "score": 0.9}]
    chat_api.save_message(url, cid, "user", prompt)
    chat_api.save_message(
        url, cid, "assistant", answer, sources=sources,
        route="rag", latency_ms=5, cache_hit=False,
    )
    return {
        "answer": answer,
        "sources": sources,
        "refused": False,
        "prompt_versions": {"system": 1, "answer": 1},
        "route": "rag",
        "latency_ms": 5,
        "cache_hit": False,
        "weather": None,
        "conversation_id": cid,
        "rewritten": False,
        "rewritten_question": prompt,
    }


def test_app_boots_with_chat_input(ui_env):
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    assert at.exception == []
    assert len(at.chat_input) == 1


def test_missing_config_shows_friendly_error(ui_env, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY")
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    assert at.exception == []
    assert any("NVIDIA_API_KEY" in e.value for e in at.error)


def test_send_message_flow(ui_env, monkeypatch):
    monkeypatch.setattr("rag.chat.ask_with_history", _fake_answer)
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    assert at.exception == []
    at.chat_input[0].set_value("hello alpha").run()
    assert at.exception == []
    assert any("CANNED reply" in m.value for m in at.markdown)


def test_backend_error_is_friendly(ui_env, monkeypatch):
    def boom(prompt, **kwargs):
        raise NimAuthError("Missing NVIDIA_API_KEY")

    monkeypatch.setattr("rag.chat.ask_with_history", boom)
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    at.chat_input[0].set_value("hi").run()
    assert at.exception == []
    assert any("credentials" in e.value for e in at.error)


def test_feedback_rename_delete_flow(ui_env):
    conv = chat_api.create_conversation(ui_env, title="t009-flow")
    chat_api.save_message(ui_env, conv["id"], "user", "q")
    chat_api.save_message(
        ui_env, conv["id"], "assistant", "a",
        sources=[{"doc_name": "d", "chunk_index": 0, "score": 1.0}],
        route="rag", latency_ms=3, cache_hit=False,
    )
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    at.session_state["conv_id"] = conv["id"]
    at.run()
    assert at.exception == []
    assert len(at.feedback) == 1
    at.feedback[0].set_value(1).run()
    assert at.exception == []
    got = chat_api.get_conversation(ui_env, conv["id"])
    assert got["messages"][1]["feedback"] == 1
    assert len(at.expander) >= 1  # sources expander present
    at.text_input[0].set_value("t009-renamed").run()
    rename = next(b for b in at.button if b.label == "Rename")
    rename.click().run()
    assert chat_api.get_conversation(ui_env, conv["id"])["title"] == (
        "t009-renamed"
    )
    assert len(at.download_button) == 1  # export available
    delete = next(b for b in at.button if b.label == "Delete chat")
    delete.click().run()
    with pytest.raises(chat_api.HistoryError):
        chat_api.get_conversation(ui_env, conv["id"])


def test_pages_boot_without_errors(ui_env):
    for page in ("1_Upload", "2_History", "3_Settings"):
        at = AppTest.from_file(str(ROOT / "pages" / f"{page}.py")).run()
        assert at.exception == [], page


def test_settings_clear_cache_button(ui_env):
    at = AppTest.from_file(str(ROOT / "pages" / "3_Settings.py")).run()
    assert at.exception == []
    clear = next(b for b in at.button if b.label == "Clear all caches")
    clear.click().run()
    assert at.exception == []
    assert any("Caches cleared" in m.value for m in at.success)
