"""Session manager: isolation, concurrency, locks (§3.5).

Isolation without VMs: each session is an asyncio task with its own ``thread_id``
and RunContext. A global semaphore bounds concurrent runs; a per-session lock
serializes writes to one thread.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from ..config import max_concurrent


@dataclass
class Session:
    session_id: str
    thread_id: str
    metadata: dict = field(default_factory=dict)


class SessionManager:
    def __init__(self, max_concurrent_runs: int | None = None) -> None:
        self._max = max_concurrent_runs or max_concurrent()
        self._sem = asyncio.Semaphore(self._max)
        self._locks: dict[str, asyncio.Lock] = {}
        self._sessions: dict[str, Session] = {}

    async def acquire(self, session_id: str | None = None) -> Session:
        """Acquire a global concurrency slot and return (or create) a Session."""
        await self._sem.acquire()
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        session = self._sessions.get(sid)
        if session is None:
            session = Session(session_id=sid, thread_id=sid)
            self._sessions[sid] = session
        return session

    async def release(self, session: Session) -> None:
        self._sem.release()

    def lock(self, session_id: str) -> asyncio.Lock:
        """One active run per session."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]
