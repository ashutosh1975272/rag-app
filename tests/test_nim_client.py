"""NIM client tests: injected fakes for retry/auth/roles + live probe.

Mocked tests never touch the network (pydantic forbids patching the real
LangChain objects, so fakes are injected through the documented seams).
Live tests run only with NVIDIA_API_KEY + NIM_CHAT_MODEL + NIM_EMBED_MODEL.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag.config import ConfigError
from rag.nim_client import NimAuthError, NimClient, NimError

DUMMY_KEY = "test-key-123"
CHAT_MODEL = "test-chat-model"
EMBED_MODEL = "test-embed-model"


class FakeEmbeddings:
    def __init__(
        self,
        documents_fn: Callable | None = None,
        query_fn: Callable | None = None,
    ) -> None:
        self._documents_fn = documents_fn or (lambda texts: [[0.0]] * len(texts))
        self._query_fn = query_fn or (lambda text: [0.0])

    def embed_documents(self, texts):
        return self._documents_fn(texts)

    def embed_query(self, text):
        return self._query_fn(text)


class FakeChat:
    def __init__(self, invoke_fn: Callable) -> None:
        self._invoke_fn = invoke_fn

    def invoke(self, messages):
        return self._invoke_fn(messages)


class FakeResult:
    def __init__(self, content) -> None:
        self.content = content


def _client(**overrides):
    args = {
        "api_key": DUMMY_KEY,
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
    }
    args.update(overrides)
    return NimClient(**args)


def test_missing_key_is_auth_error(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(NimAuthError, match="NVIDIA_API_KEY"):
        NimClient(
            api_key=None,
            chat_model=CHAT_MODEL,
            embed_model=EMBED_MODEL,
        )


def test_missing_models_rejected_fail_closed():
    with pytest.raises(ConfigError, match="NIM_CHAT_MODEL"):
        NimClient(api_key=DUMMY_KEY, embed_model=EMBED_MODEL)
    with pytest.raises(ConfigError, match="NIM_EMBED_MODEL"):
        NimClient(api_key=DUMMY_KEY, chat_model=CHAT_MODEL)


def test_embed_passage_uses_documents_pathway():
    calls = []
    client = _client(
        embeddings=FakeEmbeddings(
            documents_fn=lambda texts: (
                calls.append(texts),
                [[float(len(t))] for t in texts],
            )[1]
        )
    )
    assert client.embed(["ab", "c"]) == [[2.0], [1.0]]
    assert calls == [["ab", "c"]]


def test_embed_query_uses_query_pathway():
    calls = []
    client = _client(
        embeddings=FakeEmbeddings(
            query_fn=lambda text: (calls.append(text), [1.0, 0.0])[1]
        )
    )
    assert client.embed(["hi"], input_type="query") == [[1.0, 0.0]]
    assert calls == ["hi"]


def test_embed_rejects_bad_input_type():
    with pytest.raises(ValueError, match="input_type"):
        _client().embed(["x"], input_type="sideways")


def test_retry_on_429_then_success():
    sleeps: list[float] = []
    attempts = {"n": 0}

    def flaky(texts):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("429 rate limit exceeded")
        return [[0.5]]

    client = _client(
        max_retries=3,
        retry_base_seconds=0.5,
        sleep=sleeps.append,
        embeddings=FakeEmbeddings(documents_fn=flaky),
    )
    assert client.embed(["a"]) == [[0.5]]
    assert attempts["n"] == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0] > 0  # exponential backoff, no real waiting


def test_persistent_429_retries_three_times_then_nim_error():
    sleeps: list[float] = []
    attempts = {"n": 0}

    def always_429(texts):
        attempts["n"] += 1
        raise RuntimeError("503 Service Temporarily Unavailable")

    client = _client(
        max_retries=3,
        retry_base_seconds=0.5,
        sleep=sleeps.append,
        embeddings=FakeEmbeddings(documents_fn=always_429),
    )
    with pytest.raises(NimError, match="after 4 attempts"):
        client.embed(["a"])
    assert attempts["n"] == 4  # 1 + 3 retries
    assert len(sleeps) == 3
    assert sleeps[0] < sleeps[1] < sleeps[2]


def test_auth_failure_not_retried_and_key_never_shown():
    attempts = {"n": 0}

    def auth_fail(texts):
        attempts["n"] += 1
        raise RuntimeError("401 Unauthorized: invalid api key")

    client = _client(embeddings=FakeEmbeddings(documents_fn=auth_fail))
    with pytest.raises(NimAuthError, match="NVIDIA_API_KEY") as exc:
        client.embed(["a"])
    assert attempts["n"] == 1  # no retry on auth errors
    assert DUMMY_KEY not in str(exc.value)


def test_chat_maps_roles_and_returns_content():
    seen = {}
    client = _client(
        chat=FakeChat(
            lambda messages: (
                seen.setdefault("messages", messages),
                FakeResult("hello back"),
            )[1]
        )
    )
    out = client.chat(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
            {"role": "user", "content": "yo"},
        ]
    )
    assert out == "hello back"
    kinds = [type(m) for m in seen["messages"]]
    assert kinds == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert [m.content for m in seen["messages"]] == [
        "be brief",
        "hi",
        "hey",
        "yo",
    ]


def test_chat_rejects_unknown_role():
    with pytest.raises(ValueError, match="role"):
        _client().chat([{"role": "pirate", "content": "arr"}])


@pytest.mark.live
def test_live_embed_dimension_matches_db():
    key = os.environ.get("NVIDIA_API_KEY")
    chat_model = os.environ.get("NIM_CHAT_MODEL")
    embed_model = os.environ.get("NIM_EMBED_MODEL")
    if not (key and chat_model and embed_model):
        pytest.skip("needs NVIDIA_API_KEY + NIM_CHAT_MODEL + NIM_EMBED_MODEL")
    from rag.models import EMBEDDING_DIM

    client = NimClient(chat_model=chat_model, embed_model=embed_model)
    vectors = client.embed(["dimension probe sentence"])
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIM


@pytest.mark.live
def test_live_chat_replies():
    key = os.environ.get("NVIDIA_API_KEY")
    chat_model = os.environ.get("NIM_CHAT_MODEL")
    embed_model = os.environ.get("NIM_EMBED_MODEL")
    if not (key and chat_model and embed_model):
        pytest.skip("needs NVIDIA_API_KEY + NIM_CHAT_MODEL + NIM_EMBED_MODEL")
    client = NimClient(chat_model=chat_model, embed_model=embed_model)
    out = client.chat([{"role": "user", "content": "Reply with exactly: OK"}])
    assert out.strip()
