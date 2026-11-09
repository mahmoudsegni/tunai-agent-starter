"""A reference tool-calling agent built on the TunAI API."""

from .agent import Agent, AgentResult, ToolCallRecord, Usage
from .client import RerankResult, TunAIClient
from .config import ConfigError, Settings
from .errors import TunAIError, describe_error
from .retrieval import KnowledgeBase
from .tools import Tool, ToolRegistry

__all__ = [
    "Agent",
    "AgentResult",
    "ConfigError",
    "KnowledgeBase",
    "RerankResult",
    "Settings",
    "Tool",
    "ToolCallRecord",
    "ToolRegistry",
    "TunAIClient",
    "TunAIError",
    "Usage",
    "describe_error",
]
