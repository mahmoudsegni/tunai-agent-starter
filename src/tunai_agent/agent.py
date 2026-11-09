"""The agent loop: let the model call tools until it can answer.

One turn looks like this::

    messages = [system, ...history, user]
    loop (at most max_steps times):
        reply = chat(messages, tools)
        if reply has no tool calls:   -> that is the answer, stop
        append reply                  (the assistant's tool calls)
        for each tool call:
            result = run the tool     (errors become JSON the model can read)
            append {"role": "tool", "tool_call_id": ..., "content": result}
    out of steps -> one last call with tool_choice="none" to force an answer

The model decides *what* to do; this loop decides *whether it may*, runs it, and
bounds the cost with ``max_steps``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .client import TunAIClient
from .tools import ToolRegistry

EventHandler = Callable[[str, dict[str, Any]], None]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    def add(self, usage: Any) -> None:
        self.requests += 1
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: str
    result: str
    duration_ms: int


@dataclass
class AgentResult:
    answer: str
    steps: int
    """Model requests made, including the forced final one."""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped_early: bool = False
    """True when ``max_steps`` ran out and the answer was forced."""


class Agent:
    """A tool-calling agent on one TunAI chat model.

    Args:
        client: The TunAI client.
        tools: The tools the model may call.
        system_prompt: Role, rules and style. See ``prompts.py``.
        model: Chat model id. Defaults to ``TUNAI_CHAT_MODEL``. It must support
            function tools on /v1/chat/completions.
        max_steps: Upper bound on model requests per turn. Each tool round trip
            is one; the bound is what keeps a confused model from looping forever
            on your balance.
        max_history_messages: How much earlier conversation to resend.
        on_event: Optional callback ``(event, data)`` for logging and UIs.
            Events: ``model_request``, ``tool_call``, ``tool_result``, ``answer``.
    """

    def __init__(
        self,
        client: TunAIClient,
        tools: ToolRegistry,
        *,
        system_prompt: str,
        model: str | None = None,
        max_steps: int = 6,
        max_history_messages: int = 20,
        on_event: EventHandler | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.model = model or client.settings.chat_model
        self.max_steps = max_steps
        self.max_history_messages = max_history_messages
        self._emit = on_event or (lambda event, data: None)

    def run(self, user_message: str, history: Sequence[dict[str, Any]] = ()) -> AgentResult:
        """Answer one user message.

        ``history`` holds earlier ``user``/``assistant`` text turns. Tool calls
        from earlier turns are not resent: the final answers already carry what
        they found, and replaying every tool transcript would grow each request
        without bound.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *self._recent(history),
            {"role": "user", "content": user_message},
        ]
        result = AgentResult(answer="", steps=0)
        schemas = self.tools.schemas()

        for _ in range(self.max_steps):
            message = self._request(messages, result, tools=schemas)
            if not message.tool_calls:
                return self._finish(result, message.content)

            messages.append(_assistant_message(message))
            for call in message.tool_calls:
                messages.append(self._run_tool(call, result))

        # Out of steps: keep the tools declared (the transcript references them)
        # but forbid calling them, so the model must answer with what it has.
        result.stopped_early = True
        message = self._request(messages, result, tools=schemas, tool_choice="none")
        return self._finish(result, message.content)

    def _recent(self, history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        return turns[-self.max_history_messages :] if self.max_history_messages else []

    def _request(self, messages: list[dict[str, Any]], result: AgentResult, **kwargs: Any) -> Any:
        result.steps += 1
        self._emit("model_request", {"step": result.steps, "model": self.model})
        response = self.client.chat(messages, model=self.model, **kwargs)
        result.usage.add(response.usage)
        return response.choices[0].message

    def _run_tool(self, call: Any, result: AgentResult) -> dict[str, Any]:
        name, arguments = call.function.name, call.function.arguments
        self._emit("tool_call", {"name": name, "arguments": arguments})
        started = time.perf_counter()
        output = self.tools.execute(name, arguments)
        record = ToolCallRecord(
            name, arguments, output, int((time.perf_counter() - started) * 1000)
        )
        result.tool_calls.append(record)
        self._emit(
            "tool_result", {"name": name, "result": output, "duration_ms": record.duration_ms}
        )
        return {"role": "tool", "tool_call_id": call.id, "content": output}

    def _finish(self, result: AgentResult, content: str | None) -> AgentResult:
        result.answer = (content or "").strip()
        self._emit("answer", {"answer": result.answer, "usage": result.usage})
        return result


def _assistant_message(message: Any) -> dict[str, Any]:
    """The assistant's tool-call turn, in the shape the API expects it replayed."""
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ],
    }
