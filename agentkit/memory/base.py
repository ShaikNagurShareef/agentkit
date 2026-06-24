"""Memory interface, scopes, config, records (§5.1, §5.3)."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..context import RunContext
from ..types import Message

# session = short-term; user = per-user long-term; agent = global long-term.
Scope = Literal["session", "user", "agent"]


class MemoryConfig(BaseModel):
    backend: Literal["local", "sqlite_vec", "chroma", "qdrant", "pgvector"] = "local"
    strategies: list[Literal["semantic", "summary", "user_preference"]] = Field(
        default_factory=lambda: ["semantic", "summary"]
    )
    embedder: str = "default"
    top_k: int = 6
    write_async: bool = True  # extraction off the hot path


class MemoryRecord(BaseModel):
    id: str
    scope: Scope
    text: str
    metadata: dict = Field(default_factory=dict)
    score: float | None = None  # set on search


class Summary(BaseModel):
    session_id: str
    text: str


@runtime_checkable
class MemoryStore(Protocol):
    """One store interface for all scopes (§5.1)."""

    async def add(
        self, items: list[Message] | list[str], *, scope: Scope, ctx: RunContext
    ) -> None: ...

    async def search(
        self, query: str, *, scope: Scope, k: int = 6, ctx: RunContext | None = None
    ) -> list[MemoryRecord]: ...

    async def get_session(self, session_id: str) -> list[Message]: ...

    async def summarize(self, session_id: str) -> Summary: ...

    async def forget(self, *, scope: Scope, filter: dict) -> int: ...
