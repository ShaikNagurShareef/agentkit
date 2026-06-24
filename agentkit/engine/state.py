"""Engine state channels (§3.1.3).

AgentState is the typed channel set the compiled graph operates on. The
``messages`` channel uses an append-merge reducer so concurrent writes (e.g. from
parallel flow branches in later milestones) merge deterministically by message id.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from ..errors import ErrorInfo
from ..types import Message, ToolCall, ToolCallRecord, Usage


def add_usage(left: Usage | None, right: Usage | None) -> Usage:
    """Accumulate token/cost usage across model calls."""
    if left is None:
        return right or Usage()
    if right is None:
        return left
    return left + right


def merge_records(
    left: list[ToolCallRecord] | None, right: list[ToolCallRecord] | None
) -> list[ToolCallRecord]:
    """Append tool-call records."""
    return list(left or []) + list(right or [])


def merge_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """Append-merge reducer: new messages append; same-id messages replace in place."""
    if not left:
        return list(right)
    if not right:
        return list(left)
    by_id = {m.id: i for i, m in enumerate(left)}
    merged = list(left)
    for m in right:
        if m.id in by_id:
            merged[by_id[m.id]] = m
        else:
            merged.append(m)
            by_id[m.id] = len(merged) - 1
    return merged


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Message], merge_messages]
    session_id: str
    step: int
    scratchpad: dict[str, Any]
    pending_tool_calls: list[ToolCall]
    memory_hits: list[Any]
    status: Literal["running", "awaiting_tool", "done", "error", "interrupted"]
    error: ErrorInfo | None
    usage: Annotated[Usage, add_usage]
    tool_records: Annotated[list[ToolCallRecord], merge_records]
