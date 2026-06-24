"""AgentKit — a library-first, VM-free agent harness & orchestration platform.

M1 "Spine": import an Agent, register @tool functions, and run a tool-using task
in-process. Later milestones add serving, MCP/A2A, memory, observability, and
the Flow engine.
"""

from . import errors
from .agent import Agent
from .flow import Flow
from .models.base import ModelSettings
from .tools.base import Tool, ToolResult, tool
from .tools.registry import DEFAULT_REGISTRY, ToolRegistry
from .types import Message, RunResult, ToolCall, ToolCallRecord, Usage

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Flow",
    "tool",
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "DEFAULT_REGISTRY",
    "ModelSettings",
    "Message",
    "RunResult",
    "ToolCall",
    "ToolCallRecord",
    "Usage",
    "errors",
    "__version__",
]
