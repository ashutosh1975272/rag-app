"""Message router: rules first, LLM fallback, always logged.

`route()` returns `{"route": rag|weather|chitchat, "reason": ..., "city":
...}`. Weather matches also attempt a city extraction for the weather
tool; `city` is None when no city could be parsed.
"""

from __future__ import annotations

import logging
import re

from rag.prompts import get_active, seed_defaults

logger = logging.getLogger(__name__)

WEATHER_RE = re.compile(
    r"\b(weather|forecast|temperature|temp|rain|raining|sunny|cloudy|"
    r"humidity|windy|storm|stormy|snow|snowing)\b",
    re.IGNORECASE,
)

GREETINGS = frozenset(
    [
        "hi", "hello", "hey", "yo", "thanks", "thank", "thankyou",
        "bye", "goodbye", "morning", "afternoon", "evening",
    ]
)
CITY_AFTER_PREP = re.compile(
    r"(?:weather|forecast|temperature|temp|rain)\s+"
    r"(?:in|at|for|near)\s+"
    r"([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,2})"
)
CITY_SINGLE_WORD = re.compile(
    r"(?:weather|forecast|temperature|temp|rain)\s+"
    r"(?:in|at|for|near)\s+([A-Za-z][\w.'-]*)"
)

VALID_ROUTES = ("rag", "weather", "chitchat")


def _extract_city(message: str) -> str | None:
    match = CITY_AFTER_PREP.search(message)
    if match:
        return match.group(1).strip().rstrip("?!.").strip() or None
    single = CITY_SINGLE_WORD.search(message)
    if single:
        return single.group(1).strip().rstrip("?!.").strip() or None
    return None


def _is_greeting(message: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", message.lower())
    if not words or len(words) > 4:
        return False
    first = words[0].rstrip("'")
    if first in GREETINGS:
        return True
    return words[:2] == ["good", "morning"] or words[:2] == [
        "good",
        "afternoon",
    ] or words[:2] == ["good", "evening"]


def route(message: str, *, database_url: str, client) -> dict:
    """Classify *message*; never raises on ambiguous input."""
    if WEATHER_RE.search(message):
        out = {
            "route": "weather",
            "reason": "rule:weather-keyword",
            "city": _extract_city(message),
        }
    elif _is_greeting(message):
        out = {"route": "chitchat", "reason": "rule:greeting", "city": None}
    else:
        seed_defaults(database_url)
        _, template = get_active(database_url, "router")
        label = (
            client.chat(
                [{"role": "user", "content": template.format(message=message)}]
            )
            .strip()
            .lower()
            .split()
        )
        first = label[0] if label else ""
        if first in VALID_ROUTES:
            out = {"route": first, "reason": "llm", "city": None}
        else:
            out = {
                "route": "rag",
                "reason": f"llm-unparseable:{first[:20] or '?'}->rag",
                "city": None,
            }
    logger.info(
        "route=%s reason=%s city=%s msg=%r",
        out["route"],
        out["reason"],
        out["city"],
        message[:60],
    )
    return out
