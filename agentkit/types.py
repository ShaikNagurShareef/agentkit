"""Core wire/data types shared across AgentKit (§3.1.4, §3.1.5).

All public payloads are Pydantic v2 models (cross-cutting concern §2.3: Typing).
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from .errors import ErrorInfo

Role = Literal["system", "user", "assistant", "tool"]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ToolCall(BaseModel):
    """A model's request to invoke a tool."""

    id: str = Field(default_factory=lambda: _new_id("call"))
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A normalized chat message, provider-agnostic.

    The provider adapters translate to/from this shape so the engine and the
    rest of the library never see a provider's native message format.
    """

    id: str = Field(default_factory=lambda: _new_id("msg"))
    role: Role
    content: str | None = None
    # assistant -> tool calls it wants to make
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # tool -> which call this is a result for
    tool_call_id: str | None = None
    name: str | None = None  # tool name (for tool messages)


class ToolCallRecord(BaseModel):
    """An executed tool call, surfaced on the RunResult."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    content: Any = None
    error: ErrorInfo | None = None
    latency_ms: float | None = None


class Usage(BaseModel):
    """Token + cost accounting for a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=(
                None
                if self.cost_usd is None and other.cost_usd is None
                else (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
            ),
        )


class RunResult(BaseModel):
    """The result of one Agent/Flow invocation (§3.1.5)."""

    output: str = ""
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    session_id: str
    status: Literal["done", "error", "interrupted"] = "done"
    usage: Usage = Field(default_factory=Usage)
    trace_url: str | None = None
    error: ErrorInfo | None = None


# --- Streaming events (stream=True yields these) -------------------------------


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class ToolStartEvent(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolEndEvent(BaseModel):
    type: Literal["tool_end"] = "tool_end"
    id: str
    name: str
    ok: bool
    content: Any = None


class StepEvent(BaseModel):
    type: Literal["step"] = "step"
    step: int


class NodeStartEvent(BaseModel):
    type: Literal["node_start"] = "node_start"
    id: str
    name: str
    kind: str = "node"  # agent | function | flow | merge | condition


class NodeEndEvent(BaseModel):
    type: Literal["node_end"] = "node_end"
    id: str
    name: str
    output: str = ""
    ok: bool = True


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    result: RunResult


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: ErrorInfo


RunEvent = Union[
    TokenEvent,
    ToolStartEvent,
    ToolEndEvent,
    StepEvent,
    NodeStartEvent,
    NodeEndEvent,
    DoneEvent,
    ErrorEvent,
]
