"""Discovery-probe helper tests (pure; no network, no keys)."""

from rag.probe import check_configured, suggest_alternatives

MODELS = [
    "01-ai/yi-large",
    "meta/llama-3.2-11b-vision-instruct",
    "nvidia/nemotron-3-embed-1b",
    "nvidia/llama-nemotron-embed-vl-1b-v2",
    "mistralai/mixtral-8x22b-instruct-v0.1",
]


def test_check_configured_all_present():
    assert check_configured(
        MODELS,
        "meta/llama-3.2-11b-vision-instruct",
        "nvidia/nemotron-3-embed-1b",
    ) == {"count": 5, "chat_ok": True, "embed_ok": True}


def test_check_configured_reports_missing():
    assert check_configured(MODELS, "gone/chat", "gone/embed") == {
        "count": 5,
        "chat_ok": False,
        "embed_ok": False,
    }


def test_suggest_alternatives_lists_embed_and_instruct():
    sug = suggest_alternatives(MODELS)
    assert sug["embed"] == [
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        "nvidia/nemotron-3-embed-1b",
    ]
    assert "mistralai/mixtral-8x22b-instruct-v0.1" in sug["chat"]
    assert "01-ai/yi-large" not in sug["chat"]
    assert "01-ai/yi-large" not in sug["embed"]
