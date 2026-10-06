"""Weather tests: mocked HTTP, cache, errors, chat wiring, live probe."""

from __future__ import annotations

import json

import psycopg
import pytest
import requests

from rag import chat as rag_chat
from rag.tools_weather import UnknownCity, WeatherError, get_weather

GEOCODE_FIXTURE = {
    "results": [
        {
            "name": "Pune",
            "country": "India",
            "latitude": 18.52,
            "longitude": 73.86,
        }
    ]
}
FORECAST_FIXTURE = {
    "current": {
        "temperature_2m": 28.5,
        "weather_code": 1,
        "wind_speed_10m": 12.3,
    }
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _mock_get_factory(calls, geocode=None, forecast=None, error=None):
    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if error is not None:
            raise error
        if "geocoding" in url:
            return FakeResponse(
                geocode if geocode is not None else GEOCODE_FIXTURE
            )
        return FakeResponse(
            forecast if forecast is not None else FORECAST_FIXTURE
        )

    return fake_get


@pytest.fixture
def no_tool_cache(migrated_db):
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        conn.execute("DELETE FROM tool_cache WHERE key LIKE 'weather:%'")
    yield migrated_db
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        conn.execute("DELETE FROM tool_cache WHERE key LIKE 'weather:%'")


def test_get_weather_parses_and_caches(no_tool_cache, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        requests, "get", _mock_get_factory(calls)
    )
    first = get_weather("Pune", database_url=no_tool_cache)
    assert first == {
        "city": "Pune",
        "country": "India",
        "temp_c": 28.5,
        "wind_kph": 12.3,
        "code": 1,
        "description": "Mainly clear",
        "cached": False,
    }
    assert len(calls) == 2  # geocode + forecast
    second = get_weather("pune", database_url=no_tool_cache)
    assert second["cached"] is True
    assert second["temp_c"] == 28.5
    assert len(calls) == 2  # served from cache, no new HTTP


def test_unknown_city_friendly(no_tool_cache, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        requests, "get", _mock_get_factory(calls, geocode={"results": []})
    )
    with pytest.raises(UnknownCity) as exc:
        get_weather("Atlantis", database_url=no_tool_cache)
    assert "Atlantis" in str(exc.value)
    assert "couldn't find" in str(exc.value)


def test_network_error_friendly(no_tool_cache, monkeypatch):
    def boom(url, params=None, timeout=None):
        raise requests.ConnectionError("dns down")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(WeatherError, match="unreachable"):
        get_weather("Pune", database_url=no_tool_cache)


def test_ask_with_history_weather_branch(no_tool_cache, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(requests, "get", _mock_get_factory(calls))

    class ChatStub:
        def chat(self, messages):
            raise AssertionError("LLM must not be called on weather path")

    res = rag_chat.ask_with_history(
        "weather in Pune", database_url=no_tool_cache, client=ChatStub()
    )
    try:
        assert res["route"] == "weather"
        assert res["refused"] is False
        assert "Pune" in res["answer"] and "28.5" in res["answer"]
        assert res["weather"]["country"] == "India"
        assert res["sources"] == []
        conv = rag_chat.get_conversation(
            no_tool_cache, res["conversation_id"]
        )
        assert [m["route"] for m in conv["messages"]] == [None, "weather"]
    finally:
        with psycopg.connect(no_tool_cache, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM conversations WHERE id = %s",
                (res["conversation_id"],),
            )


def test_ask_with_history_chitchat_branch(no_tool_cache):
    class ChatStub:
        def chat(self, messages):
            return "Hello! How can I help?"

    res = rag_chat.ask_with_history(
        "hi", database_url=no_tool_cache, client=ChatStub()
    )
    try:
        assert res["route"] == "chitchat"
        assert "Hello" in res["answer"]
        assert res["sources"] == []
    finally:
        with psycopg.connect(no_tool_cache, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM conversations WHERE id = %s",
                (res["conversation_id"],),
            )


def test_ask_with_history_unknown_city_answer(no_tool_cache, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        requests, "get", _mock_get_factory(calls, geocode={"results": []})
    )

    class ChatStub:
        def chat(self, messages):
            raise AssertionError("no LLM on unknown-city path")

    res = rag_chat.ask_with_history(
        "weather in Atlantis", database_url=no_tool_cache, client=ChatStub()
    )
    try:
        assert res["route"] == "weather"
        assert "Atlantis" in res["answer"]
        assert res.get("weather") is None
    finally:
        with psycopg.connect(no_tool_cache, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM conversations WHERE id = %s",
                (res["conversation_id"],),
            )


@pytest.mark.live
def test_live_weather_pune(migrated_db):
    res = get_weather("Pune", database_url=migrated_db)
    assert res["country"] == "India"
    assert isinstance(res["temp_c"], (int, float))
    assert isinstance(res["wind_kph"], (int, float))
    print(f"\nlive weather: {json.dumps(res, default=str)}")
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        conn.execute("DELETE FROM tool_cache WHERE key LIKE 'weather:%'")
