"""Durable state persistence (§3.4).

The public seam is the ``Checkpointer`` protocol from the design. M1 backs it with
LangGraph's own checkpoint savers, which already implement durable checkpoints +
pending writes (the design's ``checkpoint_writes`` concept) and exactly-once
resume — so we don't duplicate that machinery.

Defaults:
  * ``MemorySaver`` — in-process, durable across runs within one process. Used by
    default so an Agent runs with zero infra (NFR-1).
  * ``SqliteCheckpointer`` — file-backed durability via LangGraph's async SQLite
    saver, for when runs span processes. Opt in by passing it to the Agent.

A custom-schema backend matching §3.4's exact tables can register later via the
``agentkit.backends`` entry point.
"""

from __future__ import annotations

from typing import Any, Protocol

from langgraph.checkpoint.memory import MemorySaver

from ..config import db_url


class Checkpointer(Protocol):
    """Design-level checkpointer surface (§3.4).

    For M1 the concrete saver is a LangGraph ``BaseCheckpointSaver``; this protocol
    documents the intended public API that later milestones / custom backends
    implement.
    """

    async def put(self, thread_id: str, cp: Any) -> str: ...
    async def get(self, thread_id: str, checkpoint_id: str | None = None) -> Any: ...
    async def list(self, thread_id: str, limit: int = 50) -> list[Any]: ...
    async def delete(self, thread_id: str) -> None: ...


def default_saver() -> Any:
    """Return the default in-process saver (LangGraph MemorySaver)."""
    return MemorySaver()


def _sqlite_path_from_dsn(dsn: str) -> str:
    """Extract a filesystem path (or ``:memory:``) from a sqlite DSN."""
    if dsn.startswith("sqlite:///"):
        return dsn[len("sqlite:///") :]
    if dsn.startswith("sqlite://"):
        return dsn[len("sqlite://") :] or ":memory:"
    return dsn


class SqliteCheckpointer:
    """File-backed checkpointer builder over LangGraph's async SQLite saver.

    Use under a persistent event loop (e.g. the M2 serve runtime). Returns an
    async context manager yielding a LangGraph saver ready to attach at compile.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.path = _sqlite_path_from_dsn(dsn or db_url())

    def saver(self) -> Any:
        """Return the LangGraph AsyncSqliteSaver context manager for ``self.path``."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        return AsyncSqliteSaver.from_conn_string(self.path)
