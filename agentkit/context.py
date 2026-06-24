"""RunContext propagation (§2.3).

A RunContext threads through engine -> tools -> protocols via contextvars so that
session id, trace id, deadline, and the secret resolver are available without
being passed explicitly through every call.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator
from contextlib import contextmanager


@dataclass
class RunContext:
    session_id: str
    trace_id: str | None = None
    # Absolute monotonic deadline (time.monotonic seconds); None = no deadline.
    deadline: float | None = None
    # Pluggable secret lookup (SecretProvider.get); default: env-backed at call site.
    secrets: Callable[[str], str | None] | None = None
    metadata: dict = field(default_factory=dict)

    def remaining_s(self) -> float | None:
        if self.deadline is None:
            return None
        return self.deadline - time.monotonic()

    def expired(self) -> bool:
        rem = self.remaining_s()
        return rem is not None and rem <= 0


_current: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "agentkit_run_context", default=None
)


def current_context() -> RunContext | None:
    return _current.get()


@contextmanager
def use_context(ctx: RunContext) -> Iterator[RunContext]:
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
