"""A thin client for the four TunAI endpoints the agent uses.

TunAI's developer API is OpenAI-compatible, so chat completions, embeddings and
speech-to-text go through the official ``openai`` SDK with ``base_url`` pointed
at TunAI. Rerank is not part of OpenAI's API (TunAI takes the request shape
Cohere and Jina use), so it is a plain HTTPS call with ``httpx``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .config import Settings
from .errors import TunAIError

EMBEDDING_BATCH_SIZE = 64
_RETRYABLE = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RerankResult:
    index: int
    """Position of the document in the list that was sent."""
    relevance_score: float


class TunAIClient:
    """Chat, embeddings, rerank and transcription against one TunAI API key.

    Args:
        settings: Base URL, key and default model ids.
        openai_client: Inject a pre-built client (tests use a fake).
        http_client: Inject a pre-built ``httpx.Client`` for rerank.
        max_retries: Retries on 429 and 5xx. The OpenAI SDK backs off exponentially
            and honours ``Retry-After``; rerank does the same below.
        timeout: Seconds per request. Reasoning models can take a while.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        openai_client: Any | None = None,
        http_client: httpx.Client | None = None,
        max_retries: int = 2,
        timeout: float = 120.0,
    ) -> None:
        self.settings = settings
        self.max_retries = max_retries
        self.openai = openai_client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._http = http_client or httpx.Client(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=timeout,
        )

    # -- Chat ------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **params: Any,
    ) -> ChatCompletion:
        """``POST /v1/chat/completions`` (non-streaming)."""
        kwargs: dict[str, Any] = {"model": model or self.settings.chat_model, "messages": messages}
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = tool_choice or "auto"
        kwargs.update(params)
        return self.openai.chat.completions.create(**kwargs)

    # -- Embeddings ------------------------------------------------------------

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """``POST /v1/embeddings``, batched. Returns one vector per input, in order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
            response = self.openai.embeddings.create(
                model=model or self.settings.embedding_model, input=batch
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors

    # -- Rerank ----------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: str | None = None,
        top_n: int | None = None,
        instruction: str | None = None,
    ) -> list[RerankResult]:
        """``POST /v1/rerank``. Returns results most relevant first.

        Raises:
            TunAIError: on an error response that retrying did not fix.
            ValueError: when no rerank model is configured.
        """
        model = model or self.settings.rerank_model
        if not model:
            raise ValueError("No rerank model configured (TUNAI_RERANK_MODEL is empty).")
        body: dict[str, Any] = {"model": model, "query": query, "documents": list(documents)}
        if top_n is not None:
            body["top_n"] = top_n
        if instruction:
            body["instruction"] = instruction

        data = self._post_with_retry("/rerank", body)
        return [
            RerankResult(index=r["index"], relevance_score=float(r["relevance_score"]))
            for r in data["results"]
        ]

    def _post_with_retry(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            response = self._http.post(path, json=body)
            if response.is_success:
                return response.json()
            if response.status_code not in _RETRYABLE or attempt == self.max_retries:
                raise TunAIError.from_response(response)
            time.sleep(_retry_delay(response, attempt))
        raise AssertionError("unreachable")

    # -- Speech-to-text --------------------------------------------------------

    def transcribe(
        self, audio_path: str | Path, *, model: str | None = None, language: str | None = None
    ) -> str:
        """``POST /v1/audio/transcriptions``. Returns the transcript text."""
        kwargs: dict[str, Any] = {"model": model or self.settings.transcription_model}
        if language:
            kwargs["language"] = language
        with open(audio_path, "rb") as audio:
            result = self.openai.audio.transcriptions.create(file=audio, **kwargs)
        return result.text

    # -- Models ----------------------------------------------------------------

    def list_models(self) -> list[str]:
        """``GET /v1/models``. Every model id this key can call."""
        return [model.id for model in self.openai.models.list()]

    def close(self) -> None:
        self._http.close()
        close = getattr(self.openai, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> TunAIClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return min(0.5 * 2**attempt, 8.0)
