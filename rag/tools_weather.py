"""Weather tool: Open-Meteo geocode + forecast (free, no key).

Results are cached ~10 minutes through the layer-4 tool cache. City
ambiguity ("Paris") resolves to the top geocode hit with its country shown.
"""

from __future__ import annotations

import requests

from rag.toolcache import tool_get, tool_set

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 15
TTL_SECONDS = 600

WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherError(RuntimeError):
    """Friendly weather failure (no tracebacks to callers)."""


class UnknownCity(WeatherError):
    """Geocoding found no matching place."""


def describe_code(code: int) -> str:
    """Human label for a WMO weather code."""
    return WMO_DESCRIPTIONS.get(code, f"Code {code}")


def _get_json(url: str, params: dict) -> dict:
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise WeatherError(
            "Weather service unreachable right now; please try again."
        ) from exc
    if not isinstance(data, dict):
        raise WeatherError("Weather service returned an unexpected response.")
    return data


def get_weather(
    city: str,
    *,
    database_url: str,
    redis_url: str | None = None,
    ttl_seconds: int = TTL_SECONDS,
) -> dict:
    """Current weather for *city* (cached; `cached` reports the hit)."""
    key = f"weather:{city.strip().lower()}"
    cached = tool_get(database_url, key, redis_url=redis_url)
    if cached is not None:
        return {**cached, "cached": True}

    geo = _get_json(
        GEOCODE_URL,
        {"name": city.strip(), "count": 1, "language": "en", "format": "json"},
    )
    results = geo.get("results") or []
    if not results:
        raise UnknownCity(
            f"I couldn't find a place called '{city.strip()}'. "
            "Check the spelling and try again."
        )
    place = results[0]
    try:
        forecast = _get_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m",
            },
        )
        current = forecast["current"]
        payload = {
            "city": place["name"],
            "country": place.get("country", ""),
            "temp_c": float(current["temperature_2m"]),
            "wind_kph": float(current["wind_speed_10m"]),
            "code": int(current["weather_code"]),
            "description": describe_code(int(current["weather_code"])),
            "cached": False,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherError(
            "Weather service returned an unexpected response."
        ) from exc
    tool_set(
        database_url, key, payload,
        ttl_seconds=ttl_seconds, redis_url=redis_url,
    )
    return payload
