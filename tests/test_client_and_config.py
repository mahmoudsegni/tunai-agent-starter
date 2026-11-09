import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from tunai_agent.client import TunAIClient
from tunai_agent.config import ConfigError, Settings
from tunai_agent.errors import TunAIError, describe_error

from .conftest import TEST_KEY


def _client(settings: Settings, handler, openai_client=None) -> TunAIClient:
    http = httpx.Client(
        base_url=settings.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {settings.api_key}"},
    )
    return TunAIClient(settings, openai_client=openai_client or object(), http_client=http)


def _error(status: int, code: str, message: str = "nope", headers=None) -> httpx.Response:
    body = {"error": {"message": message, "type": "x", "param": None, "code": code}}
    return httpx.Response(status, json=body, headers={"x-request-id": "req_123", **(headers or {})})


class TestRerank:
    def test_sends_the_cohere_shape_and_parses_results(self, settings):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9}]})

        results = _client(settings, handler).rerank("q", ["a", "b"], top_n=1, instruction="find")

        assert seen["path"] == "/v1/rerank"
        assert seen["auth"] == f"Bearer {TEST_KEY}"
        assert seen["body"] == {
            "model": "Qwen/Qwen3-Reranker-4B",
            "query": "q",
            "documents": ["a", "b"],
            "top_n": 1,
            "instruction": "find",
        }
        assert results[0].index == 1 and results[0].relevance_score == 0.9

    def test_retries_a_503_then_succeeds(self, settings, monkeypatch):
        monkeypatch.setattr("tunai_agent.client.time.sleep", lambda s: None)
        responses = iter([_error(503, "upstream_error"), httpx.Response(200, json={"results": []})])
        assert _client(settings, lambda r: next(responses)).rerank("q", ["a"]) == []

    def test_does_not_retry_a_402(self, settings):
        calls = []

        def handler(request):
            calls.append(request)
            return _error(402, "insufficient_quota", "Insufficient balance.")

        with pytest.raises(TunAIError) as info:
            _client(settings, handler).rerank("q", ["a"])
        assert len(calls) == 1
        assert (info.value.status, info.value.code, info.value.request_id) == (
            402,
            "insufficient_quota",
            "req_123",
        )

    def test_requires_a_model(self):
        client = _client(Settings(api_key=TEST_KEY, rerank_model=None), lambda r: None)
        with pytest.raises(ValueError):
            client.rerank("q", ["a"])


def test_embed_batches_and_keeps_input_order(settings, monkeypatch):
    monkeypatch.setattr("tunai_agent.client.EMBEDDING_BATCH_SIZE", 2)
    batches = []

    def create(model, input):
        batches.append(input)
        # Returned out of order on purpose: the client must sort by index.
        data = [SimpleNamespace(index=i, embedding=[float(len(t))]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))

    fake_openai = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    vectors = _client(settings, lambda r: None, fake_openai).embed(["a", "bb", "ccc"])

    assert batches == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0], [2.0], [3.0]]


class TestDescribeError:
    def test_budget_and_balance_402s_get_different_advice(self):
        budget = describe_error(TunAIError(402, "Budget reached.", code="budget_exceeded"))
        balance = describe_error(
            TunAIError(402, "Insufficient balance.", code="insufficient_quota")
        )
        assert "monthly budget" in budget
        assert "Top up" in balance

    def test_rate_limit_reads_the_headers(self):
        headers = httpx.Headers(
            {"x-ratelimit-limit-requests": "60", "x-ratelimit-reset-requests": "12s"}
        )
        message = describe_error(
            TunAIError(429, "Slow down.", "rate_limit_exceeded", headers=headers)
        )
        assert "60 requests per minute" in message and "12s" in message

    def test_openai_sdk_errors_include_code_and_request_id(self):
        response = _error(404, "model_not_found", "The model 'x' does not exist.")
        response.request = httpx.Request("POST", "https://api.tunai.tn/v1/chat/completions")
        exc = openai.NotFoundError("not found", response=response, body=response.json()["error"])
        message = describe_error(exc)
        assert message.startswith("[404 model_not_found] The model 'x' does not exist.")
        assert "req_123" in message and "tunai-agent models" in message


class TestSettings:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        for name in ("TUNAI_API_KEY", "TUNAI_RERANK_MODEL", "TUNAI_BASE_URL"):
            monkeypatch.delenv(name, raising=False)

    def test_missing_key_explains_where_to_get_one(self):
        with pytest.raises(ConfigError, match=r"console\.tunai\.tn"):
            Settings.from_env(env_file=None)

    def test_rejects_a_key_from_another_provider(self, monkeypatch):
        monkeypatch.setenv("TUNAI_API_KEY", "sk-proj-abc")
        with pytest.raises(ConfigError, match="sk-tunai-"):
            Settings.from_env(env_file=None)

    def test_key_never_appears_in_repr(self, monkeypatch):
        monkeypatch.setenv("TUNAI_API_KEY", TEST_KEY)
        settings = Settings.from_env(env_file=None)
        assert TEST_KEY not in repr(settings)
        assert settings.masked_key == "sk-tunai-…0000"

    def test_empty_rerank_model_disables_reranking(self, monkeypatch):
        monkeypatch.setenv("TUNAI_API_KEY", TEST_KEY)
        monkeypatch.setenv("TUNAI_RERANK_MODEL", "")
        monkeypatch.setenv("TUNAI_BASE_URL", "http://localhost:8000/v1/")
        settings = Settings.from_env(env_file=None)
        assert settings.rerank_model is None
        assert settings.base_url == "http://localhost:8000/v1"
