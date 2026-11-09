"""Look up an order. Stands in for your real backend: a database, an ERP, an HTTP API.

The demo reads ``data/orders.json``. To connect a real system, keep the tool's
name, description and parameters, and replace ``OrderStore.get``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import Tool

_ORDER_ID = re.compile(r"^DJ-\d{4}$")


def normalise_order_id(raw: str) -> str:
    """Accept what customers actually type: 'dj 1002', '#1002', 'DJ-1002'."""
    compact = re.sub(r"[\s#]", "", raw).upper()
    if compact.isdigit():
        compact = f"DJ-{compact}"
    elif compact.startswith("DJ") and not compact.startswith("DJ-"):
        compact = f"DJ-{compact[2:]}"
    return compact


class OrderStore:
    """Read-only access to orders, keyed by order id."""

    def __init__(self, orders: dict[str, dict[str, Any]]) -> None:
        self._orders = orders

    @classmethod
    def from_json(cls, path: str | Path) -> OrderStore:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({record["order_id"]: record for record in records})

    def get(self, order_id: str) -> dict[str, Any] | None:
        return self._orders.get(order_id)


def make_order_tool(store: OrderStore) -> Tool:
    def get_order_status(order_id: str) -> dict[str, Any]:
        normalised = normalise_order_id(order_id)
        if not _ORDER_ID.match(normalised):
            return {"error": f"'{order_id}' is not an order number. They look like DJ-1002."}
        order = store.get(normalised)
        if order is None:
            return {"error": f"no order {normalised}. Ask the customer to check the number."}
        return order

    return Tool(
        name="get_order_status",
        description=(
            "Look up one order by its number (format DJ-1234). Returns status, items with "
            "prices in TND, shipping governorate, carrier, tracking number and estimated "
            "delivery date. Never guess order details: call this instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order number as the customer gave it, e.g. 'DJ-1002'.",
                }
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
        handler=get_order_status,
    )
