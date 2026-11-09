import json

import pytest

from tunai_agent.agent import Agent
from tunai_agent.tools import Tool, ToolRegistry

from .conftest import ScriptedChatClient, reply, tool_call


def _registry(calls: list) -> ToolRegistry:
    def lookup(order_id: str) -> dict:
        calls.append(order_id)
        return {"order_id": order_id, "status": "shipped"}

    def broken() -> dict:
        raise RuntimeError("database is down")

    return ToolRegistry(
        [
            Tool(
                name="get_order_status",
                description="Look up an order.",
                parameters={
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
                handler=lookup,
            ),
            Tool("broken", "Always fails.", {"type": "object", "properties": {}}, broken),
        ]
    )


def test_answers_directly_when_no_tool_is_needed(settings):
    client = ScriptedChatClient(settings, [reply("Ahla! How can I help?")])
    agent = Agent(client, _registry([]), system_prompt="sys")

    result = agent.run("Hi")

    assert result.answer == "Ahla! How can I help?"
    assert result.steps == 1
    assert result.tool_calls == []
    assert client.requests[0]["messages"][0] == {"role": "system", "content": "sys"}
    assert len(client.requests[0]["tools"]) == 2


def test_runs_a_tool_and_feeds_the_result_back(settings):
    calls: list = []
    client = ScriptedChatClient(
        settings,
        [
            reply(tool_calls=[tool_call("get_order_status", {"order_id": "DJ-1002"}, "call_a")]),
            reply("Your order DJ-1002 has shipped."),
        ],
    )
    agent = Agent(client, _registry(calls), system_prompt="sys")

    result = agent.run("Where is DJ-1002?")

    assert calls == ["DJ-1002"]
    assert result.answer == "Your order DJ-1002 has shipped."
    assert result.steps == 2
    assert result.usage.total_tokens == 240

    second = client.requests[1]["messages"]
    assistant, tool = second[-2], second[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_a"
    assert tool == {
        "role": "tool",
        "tool_call_id": "call_a",
        "content": json.dumps({"order_id": "DJ-1002", "status": "shipped"}),
    }


def test_runs_parallel_tool_calls_in_one_step(settings):
    calls: list = []
    client = ScriptedChatClient(
        settings,
        [
            reply(
                tool_calls=[
                    tool_call("get_order_status", {"order_id": "DJ-1001"}, "a"),
                    tool_call("get_order_status", {"order_id": "DJ-1002"}, "b"),
                ]
            ),
            reply("Both found."),
        ],
    )
    result = Agent(client, _registry(calls), system_prompt="sys").run("Check both")

    assert calls == ["DJ-1001", "DJ-1002"]
    tool_messages = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    assert result.steps == 2


def test_a_failing_tool_is_reported_to_the_model_not_raised(settings):
    client = ScriptedChatClient(
        settings, [reply(tool_calls=[tool_call("broken", {})]), reply("Sorry, try later.")]
    )
    result = Agent(client, _registry([]), system_prompt="sys").run("Do it")

    tool_message = client.requests[1]["messages"][-1]
    assert json.loads(tool_message["content"]) == {"error": "RuntimeError: database is down"}
    assert result.answer == "Sorry, try later."


def test_forces_an_answer_when_max_steps_runs_out(settings):
    loop = reply(tool_calls=[tool_call("get_order_status", {"order_id": "DJ-1001"})])
    client = ScriptedChatClient(settings, [loop, loop, reply("Here is what I found.")])
    result = Agent(client, _registry([]), system_prompt="sys", max_steps=2).run("?")

    assert result.stopped_early
    assert result.steps == 3
    assert client.requests[-1]["tool_choice"] == "none"
    assert client.requests[-1]["tools"]  # still declared: the transcript references them
    assert result.answer == "Here is what I found."


def test_resends_only_recent_text_history(settings):
    client = ScriptedChatClient(settings, [reply("ok")])
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "tool", "tool_call_id": "x", "content": "{}"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]
    Agent(client, _registry([]), system_prompt="sys", max_history_messages=2).run("five", history)

    sent = client.requests[0]["messages"]
    assert [m["content"] for m in sent] == ["sys", "three", "four", "five"]


def test_emits_events_for_tracing(settings):
    events: list = []
    client = ScriptedChatClient(
        settings,
        [reply(tool_calls=[tool_call("get_order_status", {"order_id": "DJ-1001"})]), reply("done")],
    )
    Agent(client, _registry([]), system_prompt="sys", on_event=lambda e, d: events.append(e)).run(
        "?"
    )

    assert events == ["model_request", "tool_call", "tool_result", "model_request", "answer"]


def test_rejects_a_non_positive_step_budget(settings):
    with pytest.raises(ValueError):
        Agent(ScriptedChatClient(settings, []), _registry([]), system_prompt="sys", max_steps=0)
