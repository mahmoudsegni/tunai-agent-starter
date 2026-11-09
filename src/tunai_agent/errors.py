"""Turn TunAI API errors into messages a person can act on.

TunAI returns errors in OpenAI's envelope::

    {"error": {"message": "...", "type": "...", "param": null, "code": "..."}}

so the OpenAI SDK raises its usual exceptions. The ``code`` field is what tells
the cases apart (a 402 can mean "top up" or "this key hit its monthly budget").
"""

from __future__ import annotations

from typing import Any

import httpx
import openai

CONSOLE_URL = "https://console.tunai.tn"


class TunAIError(RuntimeError):
    """An error response from an endpoint called without the OpenAI SDK (rerank)."""

    def __init__(
        self,
        status: int,
        message: str,
        code: str | None = None,
        request_id: str | None = None,
        headers: httpx.Headers | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.request_id = request_id
        self.headers = headers or httpx.Headers()

    @classmethod
    def from_response(cls, response: httpx.Response) -> TunAIError:
        error = _error_body(_json_or_none(response))
        return cls(
            status=response.status_code,
            message=error.get("message") or response.text[:300] or response.reason_phrase,
            code=error.get("code"),
            request_id=response.headers.get("x-request-id"),
            headers=response.headers,
        )


def _json_or_none(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _error_body(body: Any) -> dict[str, Any]:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"]
    return {}


def describe_error(exc: BaseException) -> str:
    """A one-paragraph explanation of what went wrong and what to do about it."""
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach the TunAI API. Check your connection and TUNAI_BASE_URL."
    if isinstance(exc, openai.APIStatusError):
        # The SDK hands over the inner {"message", "code", ...} object as `body`.
        body = exc.body if isinstance(exc.body, dict) else {}
        error = _error_body(body) or body
        status, headers = exc.status_code, exc.response.headers
        message = error.get("message") or str(exc)
        code = error.get("code")
    elif isinstance(exc, TunAIError):
        status, headers, message, code = exc.status, exc.headers, exc.message, exc.code
    else:
        return f"{type(exc).__name__}: {exc}"

    hint = _hint(status, code, headers)
    request_id = headers.get("x-request-id")
    suffix = f" (request id: {request_id})" if request_id else ""
    return f"[{status}{' ' + code if code else ''}] {message}{suffix}" + (
        f"\n→ {hint}" if hint else ""
    )


def _hint(status: int, code: str | None, headers: httpx.Headers) -> str | None:
    if status == 401:
        return (
            "Check TUNAI_API_KEY. If it is right, the key may be revoked or expired: "
            f"create a new one at {CONSOLE_URL}."
        )
    if status == 402 and code == "budget_exceeded":
        return f"This key reached the monthly budget you set. Raise it at {CONSOLE_URL}."
    if status == 402 and code == "promo_exhausted":
        return f"The free promo credit is used up. Create a regular key at {CONSOLE_URL}."
    if status == 402:
        return f"Your balance is too low for this request. Top up at {CONSOLE_URL}/billing."
    if status == 403 and code == "insufficient_scope":
        return "This key is restricted to other endpoints. Create a key with the scope you need."
    if status == 404 and code == "model_not_found":
        return "Run `tunai-agent models` to see every model id your key can use."
    if status == 429:
        limit = headers.get("x-ratelimit-limit-requests")
        reset = headers.get("x-ratelimit-reset-requests")
        if limit:
            return (
                f"This key allows {limit} requests per minute; it resets in {reset or 'a moment'}."
            )
        return "Too many requests. Wait a moment and retry."
    if status >= 500:
        return "TunAI or the upstream provider had a problem. Retrying usually works."
    return None
