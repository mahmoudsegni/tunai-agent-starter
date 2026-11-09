"""Offline fakes. The test suite never calls the network and needs no API key."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from tunai_agent.config import Settings

TEST_KEY = "sk-tunai-FAKE-test-key-0000"


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key=TEST_KEY)


def tool_call(
    name: str, arguments: dict[str, Any] | str, call_id: str = "call_1"
) -> SimpleNamespace:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments=raw)
    )


def reply(
    content: str | None = None, tool_calls: list[SimpleNamespace] | None = None
) -> SimpleNamespace:
    """A chat completion shaped like the OpenAI SDK's."""
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


class ScriptedChatClient:
    """Stands in for TunAIClient in agent tests: returns scripted replies in order."""

    def __init__(self, settings: Settings, replies: list[SimpleNamespace]) -> None:
        self.settings = settings
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> SimpleNamespace:
        self.requests.append({"messages": copy.deepcopy(messages), **kwargs})
        if not self.replies:
            raise AssertionError("the agent made more requests than the test scripted")
        return self.replies.pop(0)
