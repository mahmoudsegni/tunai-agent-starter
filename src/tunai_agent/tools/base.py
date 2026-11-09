"""Tools: a function the model can ask to call, plus the JSON Schema that describes it.

The model never runs anything itself. It replies with a *tool call* (a name and
JSON arguments); the agent looks the tool up here, validates the arguments,
runs the handler and sends the result back as a ``tool`` message.

Every outcome, including a failure, goes back to the model as JSON. A model
told ``{"error": "order DJ-9 not found"}`` can apologise or ask for the right
number; a model whose tool crashed the program can do nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolArgumentError(ValueError):
    """The model sent arguments that do not match the tool's schema."""


@dataclass(frozen=True)
class Tool:
    """One capability exposed to the model.

    Attributes:
        name: What the model calls it. Lowercase with underscores.
        description: When to use it and what it returns. The model reads this to
            decide, so it matters as much as the code.
        parameters: A JSON Schema object for the arguments.
        handler: Called with the validated arguments as keyword arguments. Returns
            anything JSON-serialisable.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """The entry for the ``tools`` array of a chat completion request."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check required fields and top-level types; drop fields the schema doesn't define."""
        properties: dict[str, Any] = self.parameters.get("properties", {})
        missing = [k for k in self.parameters.get("required", []) if k not in arguments]
        if missing:
            raise ToolArgumentError(f"missing required argument(s): {', '.join(missing)}")
        clean: dict[str, Any] = {}
        for key, value in arguments.items():
            spec = properties.get(key)
            if spec is None:
                continue
            expected = _JSON_TYPES.get(spec.get("type", ""))
            is_bool_for_number = isinstance(value, bool) and spec.get("type") in (
                "integer",
                "number",
            )
            if expected and (not isinstance(value, expected) or is_bool_for_number):
                raise ToolArgumentError(f"'{key}' must be of type {spec['type']}")
            if "enum" in spec and value not in spec["enum"]:
                raise ToolArgumentError(f"'{key}' must be one of {spec['enum']}")
            clean[key] = value
        return clean


class ToolRegistry:
    """The set of tools offered to the model, and the one place that runs them."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str | None) -> str:
        """Run a tool call and return its result as a JSON string, never raising.

        ``raw_arguments`` is the JSON string exactly as the model produced it.
        """
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"unknown tool '{name}'. Available: {', '.join(self._tools)}")
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return _error(f"arguments are not valid JSON: {exc.msg}")
        if not isinstance(arguments, dict):
            return _error("arguments must be a JSON object")
        try:
            result = tool.handler(**tool.validate(arguments))
        except ToolArgumentError as exc:
            return _error(f"invalid arguments: {exc}")
        except Exception as exc:  # reported to the model, see the module docstring
            return _error(f"{type(exc).__name__}: {exc}")
        return json.dumps(result, ensure_ascii=False, default=str)


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)
