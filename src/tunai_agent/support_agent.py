"""Wire the pieces into the Dar Jasmin support agent.

This is the only file that knows about the demo shop. To build your own agent,
copy it: pick your tools, write your system prompt, keep ``Agent`` as it is.
"""

from __future__ import annotations

from pathlib import Path

from .agent import Agent, EventHandler
from .client import TunAIClient
from .prompts import SYSTEM_PROMPT
from .retrieval import KnowledgeBase
from .tools import OrderStore, ToolRegistry, calculator_tool, make_knowledge_tool, make_order_tool

DEFAULT_KB_DIR = Path("knowledge_base")
DEFAULT_ORDERS = Path("data/orders.json")


def build_support_agent(
    client: TunAIClient,
    *,
    kb_dir: str | Path = DEFAULT_KB_DIR,
    orders_file: str | Path = DEFAULT_ORDERS,
    model: str | None = None,
    max_steps: int = 6,
    on_event: EventHandler | None = None,
) -> Agent:
    """A support agent with three tools: knowledge search, order lookup, calculator."""
    knowledge_base = KnowledgeBase.build(client, kb_dir)
    tools = ToolRegistry(
        [
            make_knowledge_tool(knowledge_base),
            make_order_tool(OrderStore.from_json(orders_file)),
            calculator_tool,
        ]
    )
    return Agent(
        client,
        tools,
        system_prompt=SYSTEM_PROMPT,
        model=model,
        max_steps=max_steps,
        on_event=on_event,
    )
