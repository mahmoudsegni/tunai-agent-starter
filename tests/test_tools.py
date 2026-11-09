import json
from pathlib import Path

import pytest

from tunai_agent.tools import OrderStore, Tool, ToolRegistry, calculator_tool, make_order_tool
from tunai_agent.tools.calculator import evaluate
from tunai_agent.tools.orders import normalise_order_id

ORDERS = Path(__file__).resolve().parents[1] / "data" / "orders.json"


def _echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echo.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "times": {"type": "integer"},
                "mode": {"type": "string", "enum": ["loud", "quiet"]},
            },
            "required": ["text"],
        },
        handler=lambda text, times=1, mode="quiet": {"echo": text * times, "mode": mode},
    )


def run(registry: ToolRegistry, name: str, args) -> dict:
    raw = args if isinstance(args, str) else json.dumps(args)
    return json.loads(registry.execute(name, raw))


class TestRegistry:
    def test_executes_with_validated_arguments(self):
        assert run(ToolRegistry([_echo_tool()]), "echo", {"text": "ya", "times": 2}) == {
            "echo": "yaya",
            "mode": "quiet",
        }

    def test_ignores_arguments_the_schema_does_not_define(self):
        assert run(ToolRegistry([_echo_tool()]), "echo", {"text": "a", "extra": 1})["echo"] == "a"

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ({}, "missing required argument(s): text"),
            ({"text": 3}, "'text' must be of type string"),
            ({"text": "a", "times": True}, "'times' must be of type integer"),
            ({"text": "a", "mode": "shout"}, "'mode' must be one of"),
            ("not json", "arguments are not valid JSON"),
            ("[1, 2]", "arguments must be a JSON object"),
        ],
    )
    def test_bad_arguments_become_errors_for_the_model(self, args, message):
        assert message in run(ToolRegistry([_echo_tool()]), "echo", args)["error"]

    def test_unknown_tool_lists_the_available_ones(self):
        error = run(ToolRegistry([_echo_tool()]), "nope", {})["error"]
        assert "unknown tool 'nope'" in error and "echo" in error

    def test_refuses_duplicate_names(self):
        with pytest.raises(ValueError):
            ToolRegistry([_echo_tool(), _echo_tool()])

    def test_schema_is_an_openai_function_tool(self):
        schema = _echo_tool().schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["parameters"]["required"] == ["text"]


class TestCalculator:
    @pytest.mark.parametrize(
        ("expression", "value"),
        [
            ("2 * 38.5 + 7", 84.0),
            ("120 + 3 * 14", 162),
            ("-(5 - 8) ** 2", -9),
            ("10 // 3 + 10 % 3", 4),
        ],
    )
    def test_arithmetic(self, expression, value):
        assert evaluate(expression) == value

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os')",
            "open('x')",
            "a + 1",
            "[1, 2]",
            "2 ** 1000",
            "'a' * 3",
            "1 if 1 else 2",
        ],
    )
    def test_rejects_anything_but_arithmetic(self, expression):
        with pytest.raises(ValueError):
            evaluate(expression)

    def test_tool_reports_division_by_zero_instead_of_raising(self):
        result = run(ToolRegistry([calculator_tool]), "calculate", {"expression": "1 / 0"})
        assert result["error"] == "division by zero"

    def test_tool_rounds_floats(self):
        result = run(ToolRegistry([calculator_tool]), "calculate", {"expression": "0.1 + 0.2"})
        assert result["result"] == 0.3


class TestOrders:
    @pytest.fixture
    def registry(self) -> ToolRegistry:
        return ToolRegistry([make_order_tool(OrderStore.from_json(ORDERS))])

    @pytest.mark.parametrize("raw", ["DJ-1002", "dj 1002", "#1002", "1002", "dj1002"])
    def test_normalises_what_customers_type(self, raw):
        assert normalise_order_id(raw) == "DJ-1002"

    def test_finds_an_order(self, registry):
        order = run(registry, "get_order_status", {"order_id": "dj-1002"})
        assert order["status"] == "shipped"
        assert order["shipping"]["tracking"] == "RP-77201145TN"

    def test_unknown_order(self, registry):
        assert (
            "no order DJ-9999"
            in run(registry, "get_order_status", {"order_id": "DJ-9999"})["error"]
        )

    def test_not_an_order_number(self, registry):
        assert (
            "not an order number"
            in run(registry, "get_order_status", {"order_id": "hello"})["error"]
        )
