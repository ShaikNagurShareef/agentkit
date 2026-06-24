"""Local memory store: embedded vector recall with zero infra (§5.3).

Default backend. Records are embedded with the local hashing embedder and ranked
by cosine similarity; session transcripts and rolling summaries are kept in memory.
Heavier backends (chroma/qdrant/pgvector) implement the same `MemoryStore` surface
and register via the `agentkit.backends` entry point.
"""

from __future__ import annotations

import uuid

from ..context import RunContext
from ..types import Message
from .base import MemoryConfig, MemoryRecord, MemoryStore, Scope, Summary
from .embeddings import cosine, embed


class LocalMemoryStore:
    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self._records: list[MemoryRecord] = []
        self._emb: dict[str, list[float]] = {}
        self._sessions: dict[str, list[Message]] = {}
        self._summaries: dict[str, str] = {}

    async def add(
        self, items: list[Message] | list[str], *, scope: Scope, ctx: RunContext
    ) -> None:
        for it in items:
            text = it.content if isinstance(it, Message) else str(it)
            if not text:
                continue
            rid = f"mem_{uuid.uuid4().hex[:12]}"
            rec = MemoryRecord(
                id=rid,
                scope=scope,
                text=text,
                metadata={
                    "session_id": ctx.session_id,
                    "user_id": ctx.metadata.get("user_id"),
                },
            )
            self._records.append(rec)
            self._emb[rid] = embed(text)

    async def search(
        self, query: str, *, scope: Scope, k: int = 6, ctx: RunContext | None = None
    ) -> list[MemoryRecord]:
        q = embed(query)
        user_id = ctx.metadata.get("user_id") if ctx else None
        scored: list[tuple[float, MemoryRecord]] = []
        for rec in self._records:
            if rec.scope != scope:
                continue
            if scope == "user" and user_id is not None and rec.metadata.get("user_id") != user_id:
                continue
            scored.append((cosine(q, self._emb[rec.id]), rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec.model_copy(update={"score": s}) for s, rec in scored[:k] if s > 0]

    async def recall(self, query: str, *, ctx: RunContext, k: int = 6) -> list[MemoryRecord]:
        """Search the long-term scopes (user + agent) for read-path injection."""
        hits = await self.search(query, scope="user", k=k, ctx=ctx)
        hits += await self.search(query, scope="agent", k=k, ctx=ctx)
        hits.sort(key=lambda r: r.score or 0.0, reverse=True)
        return hits[:k]

    # --- session transcript / summary -----------------------------------------

    def add_transcript(self, session_id: str, messages: list[Message]) -> None:
        self._sessions[session_id] = list(messages)

    def set_summary(self, session_id: str, text: str) -> None:
        self._summaries[session_id] = text

    async def get_session(self, session_id: str) -> list[Message]:
        return self._sessions.get(session_id, [])

    async def summarize(self, session_id: str) -> Summary:
        return Summary(session_id=session_id, text=self._summaries.get(session_id, ""))

    def snapshot(self) -> dict:
        """Introspection dump for the dashboard: records by scope + summaries."""
        scopes: dict[str, list[dict]] = {"session": [], "user": [], "agent": []}
        for rec in self._records:
            scopes.setdefault(rec.scope, []).append(
                {"id": rec.id, "text": rec.text, "metadata": rec.metadata}
            )
        return {
            "backend": self.config.backend,
            "strategies": self.config.strategies,
            "embedder": self.config.embedder,
            "top_k": self.config.top_k,
            "counts": {k: len(v) for k, v in scopes.items()},
            "scopes": scopes,
            "summaries": dict(self._summaries),
            "sessions": list(self._sessions),
        }

    async def forget(self, *, scope: Scope, filter: dict) -> int:
        before = len(self._records)
        kept: list[MemoryRecord] = []
        for rec in self._records:
            match = rec.scope == scope and all(
                rec.metadata.get(key) == val for key, val in filter.items()
            )
            if match:
                self._emb.pop(rec.id, None)
            else:
                kept.append(rec)
        self._records = kept
        return before - len(self._records)


def build_memory_store(config: MemoryConfig) -> MemoryStore:
    """Construct a memory store for the configured backend (default: local)."""
    if config.backend in ("local", "sqlite_vec", "chroma", "qdrant", "pgvector"):
        # Heavier backends register via agentkit.backends; default to local.
        return LocalMemoryStore(config)
    return LocalMemoryStore(config)
