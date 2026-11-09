"""Exact arithmetic for the model: totals, shipping fees, percentages.

Language models are unreliable at arithmetic, so prices are computed here. The
expression is parsed into a syntax tree and only numbers and arithmetic
operators are evaluated; there is no ``eval`` and no way to reach a name or a
function.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from .base import Tool

_MAX_EXPRESSION_LENGTH = 200
_MAX_EXPONENT = 100

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate(expression: str) -> float | int:
    """Evaluate an arithmetic expression.

    Raises:
        ValueError: for anything other than numbers, + - * / // % ** and parentheses.
    """
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression longer than {_MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not an arithmetic expression: {expression!r}") from exc
    return _eval(tree.body)


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent larger than {_MAX_EXPONENT}")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported element: {type(node).__name__}")


def _calculate(expression: str) -> dict[str, Any]:
    try:
        value = evaluate(expression)
    except ZeroDivisionError:
        return {"expression": expression, "error": "division by zero"}
    if isinstance(value, float):
        value = round(value, 6)
    return {"expression": expression, "result": value}


calculator_tool = Tool(
    name="calculate",
    description=(
        "Evaluate an arithmetic expression exactly. Use it for every price, total, "
        "discount or fee instead of doing the maths yourself. Supports + - * / // % ** "
        "and parentheses, e.g. '2 * 38.5 + 7'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The expression to evaluate."}
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    handler=_calculate,
)
