"""UI helper tests: pure formatting + friendly error mapping."""

from rag import ui_helpers
from rag.cache import clear_caches  # noqa: F401 (layer-5 import check)
from rag.chat import HistoryError
from rag.ingest import IngestError
from rag.nim_client import NimAuthError, NimError
from rag.prompts import PromptError
from rag.tools_weather import UnknownCity, WeatherError


def test_format_citation():
    assert (
        ui_helpers.format_citation({"doc_name": "a.txt", "chunk_index": 3})
        == "[a.txt #3]"
    )


def test_badge_text_all_parts():
    assert (
        ui_helpers.badge_text("rag", 120, True)
        == "rag · 120 ms · cache hit"
    )
    assert (
        ui_helpers.badge_text("weather", 45, False)
        == "weather · 45 ms"
    )
    assert ui_helpers.badge_text(None, None, False) == ""


def test_export_filename_sanitized():
    assert (
        ui_helpers.export_filename("Hello World?")
        == "hello-world.md"
    )
    assert (
        ui_helpers.export_filename("  A/B\\C:D*E  ")
        == "a-b-c-d-e.md"
    )
    assert ui_helpers.export_filename("") == "chat.md"


def test_friendly_error_auth():
    assert "NVIDIA_API_KEY" in ui_helpers.friendly_error(
        NimAuthError("Missing NVIDIA_API_KEY")
    )


def test_friendly_error_nim():
    assert "model service" in ui_helpers.friendly_error(
        NimError("NIM unavailable after 4 attempts: 503")
    ).lower()


def test_friendly_error_ingest_and_weather():
    assert "PDF" in ui_helpers.friendly_error(
        IngestError("x.pdf: could not parse as PDF.")
    )
    assert "Pune" in ui_helpers.friendly_error(
        UnknownCity("I couldn't find a place called 'Pune'.")
    )
    assert "unreachable" in ui_helpers.friendly_error(
        WeatherError("Weather service unreachable right now; please try again.")
    )


def test_friendly_error_history_and_prompt():
    assert "conversation" in ui_helpers.friendly_error(
        HistoryError("Unknown conversation: 9")
    )
    assert "prompt" in ui_helpers.friendly_error(
        PromptError("No active prompt template: 'answer'")
    ).lower()


def test_friendly_error_operational_and_unknown():
    import psycopg

    assert "database" in ui_helpers.friendly_error(
        psycopg.OperationalError("connection refused")
    ).lower()
    msg = ui_helpers.friendly_error(RuntimeError("weird boom"))
    assert "unexpected" in msg.lower()
    assert "weird boom" not in msg  # no internals leaked
