"""Search the shop's documentation (shipping, returns, payments, products)."""

from __future__ import annotations

from typing import Any

from ..retrieval import KnowledgeBase, passages_as_json
from .base import Tool


def make_knowledge_tool(knowledge_base: KnowledgeBase) -> Tool:
    def search_knowledge_base(query: str, top_k: int = 3) -> dict[str, Any]:
        top_k = max(1, min(top_k, 5))
        return passages_as_json(knowledge_base.search(query, top_k=top_k))

    return Tool(
        name="search_knowledge_base",
        description=(
            "Search Dar Jasmin's policies and product guides: shipping fees and delivery "
            "times, returns and refunds, payment methods, product care. Returns the most "
            "relevant passages with their source. Use it before answering any question "
            "about how the shop works, and cite the source."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A self-contained search query, e.g. 'shipping fee to Sfax'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many passages to return (1-5). Default 3.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=search_knowledge_base,
    )
