"""Router tests: rules, LLM fallback, and the 10-message spot-check."""

import logging

import pytest

from rag.router import route


class RouterStub:
    def __init__(self, label):
        self.label = label
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self.label


def test_weather_rule_with_city(migrated_db):
    out = route(
        "weather in Pune", database_url=migrated_db, client=RouterStub("rag")
    )
    assert out["route"] == "weather"
    assert out["city"] == "Pune"
    assert "rule" in out["reason"]


def test_temperature_rule_with_city(migrated_db):
    out = route(
        "What's the temperature in Berlin?",
        database_url=migrated_db,
        client=RouterStub("rag"),
    )
    assert out["route"] == "weather"
    assert out["city"] == "Berlin"


def test_greeting_rule(migrated_db):
    for msg in ("hi", "hello!", "thanks", "good morning"):
        out = route(
            msg, database_url=migrated_db, client=RouterStub("rag")
        )
        assert out["route"] == "chitchat", msg


def test_llm_fallback_for_ambiguous(migrated_db):
    stub = RouterStub("rag")
    out = route(
        "what does the report say about margins?",
        database_url=migrated_db,
        client=stub,
    )
    assert out == {"route": "rag", "reason": "llm", "city": None}
    assert stub.calls == 1


def test_llm_garbage_defaults_to_rag(migrated_db):
    out = route(
        "something ambiguous here",
        database_url=migrated_db,
        client=RouterStub("banana"),
    )
    assert out["route"] == "rag"
    assert "unparseable" in out["reason"]


def test_route_decision_logged(migrated_db, caplog):
    with caplog.at_level(logging.INFO, logger="rag.router"):
        route("hi", database_url=migrated_db, client=RouterStub("rag"))
    assert "route=chitchat" in caplog.text


SPOT_CHECK = [
    ("weather in Pune", "weather"),
    ("will it rain in London tomorrow?", "weather"),
    ("temperature in Tokyo right now", "weather"),
    ("hi", "chitchat"),
    ("hello there!", "chitchat"),
    ("thanks a lot", "chitchat"),
    ("what does the report say?", "rag"),
    ("summarize the uploaded document", "rag"),
    ("who won the 1903 physics nobel?", "rag"),
    ("compare the two bridges", "rag"),
]


@pytest.mark.parametrize("message,expected", SPOT_CHECK)
def test_router_spot_check(migrated_db, message, expected):
    out = route(
        message, database_url=migrated_db, client=RouterStub("rag")
    )
    assert out["route"] == expected, message
