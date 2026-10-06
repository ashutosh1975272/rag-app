"""NIM chat + embeddings via LangChain objects.

Chat goes through ``ChatOpenAI`` pointed at the NIM OpenAI-compatible
endpoint: ``ChatNVIDIA`` hard-routes this VLM model id to
``ai.api.nvidia.com/v1/gr/...``, which answers 451 for this key, while
the same model on ``integrate.api.nvidia.com/v1`` answers 200
(proven live 2026-10-06). Embeddings use ``NVIDIAEmbeddings``, which
already sends the right ``input_type`` (``query``/``passage``).
The wrapper adds: uniform ``embed()``/``chat()`` seams, retry with
exponential backoff on 429/5xx, and friendly errors that never
contain the API key.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from rag.config import require

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

_AUTH_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "invalid_api_key",
    "authentication",
)

_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "rate limit",
    "rate_limit",
    "timeout",
    "timed out",
    "overloaded",
    "try again",
    "temporarily",
)


class NimError(RuntimeError):
    """NIM call failed (message never contains the API key)."""


class NimAuthError(NimError):
    """NIM rejected the credentials."""


def _is_auth_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _AUTH_MARKERS)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, NimAuthError) or _is_auth_failure(exc):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _translate(exc: Exception, *, attempts: int) -> NimError:
    if isinstance(exc, NimAuthError) or _is_auth_failure(exc):
        return NimAuthError(
            "NIM rejected the credentials: check NVIDIA_API_KEY."
        )
    if _is_retryable(exc):
        return NimError(f"NIM unavailable after {attempts} attempts: {exc}")
    return NimError(str(exc))


_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


class NimClient:
    """Thin retrying wrapper over ``ChatNVIDIA`` / ``NVIDIAEmbeddings``.

    Args:
        api_key: falls back to the ``NVIDIA_API_KEY`` env var. Never logged.
        base_url: NIM OpenAI-compatible endpoint.
        chat_model: measured chat model id (MEMORY.md).
        embed_model: measured embedding model id (MEMORY.md).
        temperature: chat sampling temperature.
        max_retries: retries after the first attempt (default 3).
        retry_base_seconds: backoff multiplier (tests inject a recorder).
        sleep: tenacity sleep hook (tests record the backoff pattern).
        embeddings: override the LangChain embeddings object (test seam).
        chat: override the LangChain chat object (test seam).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        chat_model: str | None = None,
        embed_model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        embeddings: NVIDIAEmbeddings | None = None,
        chat: ChatOpenAI | None = None,
    ) -> None:
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise NimAuthError(
                "Missing NVIDIA_API_KEY: set it in the environment."
            )
        # Never fall back to the package's default models: only the measured
        # ids (MEMORY.md) are proven to work with this key and vector(2048).
        require(chat_model, "NIM_CHAT_MODEL")
        require(embed_model, "NIM_EMBED_MODEL")
        self.max_retries = max_retries
        self._chat_model_id = chat_model
        self._embed_model_id = embed_model
        self._embeddings = embeddings or NVIDIAEmbeddings(
            model=embed_model,
            base_url=base_url,
            api_key=key,
            truncate="END",
        )
        # Inner retries disabled: this wrapper owns the retry policy.
        self._chat = chat or ChatOpenAI(
            model=chat_model,
            base_url=base_url,
            api_key=key,
            temperature=temperature,
            max_retries=0,
        )
        self._retry = retry(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(
                multiplier=retry_base_seconds, min=retry_base_seconds
            ),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
            sleep=sleep,
        )

    @property
    def attempts(self) -> int:
        return self.max_retries + 1

    @property
    def chat_model(self):
        """The underlying LangChain chat model (for LCEL chains)."""
        return self._chat

    def _call(self, fn: Callable[[], object]) -> object:
        try:
            return self._retry(fn)()
        except NimAuthError:
            raise
        except Exception as exc:
            raise _translate(exc, attempts=self.attempts) from exc

    def embed(
        self, texts: Sequence[str], input_type: str = "passage"
    ) -> list[list[float]]:
        """Embed texts; ``input_type`` is ``passage`` (docs) or ``query``."""
        if input_type not in ("passage", "query"):
            raise ValueError(
                f"input_type must be 'passage' or 'query', got {input_type!r}"
            )
        items = list(texts)
        if input_type == "passage":
            return self._call(lambda: self._embeddings.embed_documents(items))  # type: ignore[return-value]
        return [  # type: ignore[return-value]
            self._call(lambda t=t: self._embeddings.embed_query(t))
            for t in items
        ]

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        """Chat completion; each message is ``{"role": ..., "content": ...}``."""
        lc_messages = []
        for msg in messages:
            try:
                cls = _ROLE_TO_MESSAGE[msg["role"]]
            except KeyError:
                raise ValueError(
                    f"role must be system/user/assistant, got {msg.get('role')!r}"
                ) from None
            lc_messages.append(cls(content=msg["content"]))
        result = self._call(lambda: self._chat.invoke(lc_messages))
        content = result.content  # type: ignore[union-attr]
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
