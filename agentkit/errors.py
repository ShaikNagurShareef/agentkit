"""Error taxonomy and policy (§10).

Every failure carries a typed name, a retry disposition, and a defined effect on
the run. No bare exceptions cross the public API (NFR-4).
"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorInfo(BaseModel):
    """Structured, serializable error payload carried by every AgentError."""

    type: str
    message: str
    retriable: bool = False
    where: str | None = None  # node / tool / server
    cause: str | None = None


class AgentError(Exception):
    """Base error; carries an ErrorInfo.

    Subclasses set ``retriable`` so the retry policy (§10.2) can be applied
    uniformly without inspecting concrete types.
    """

    retriable: bool = False

    def __init__(
        self,
        message: str,
        *,
        where: str | None = None,
        cause: str | None = None,
    ) -> None:
        super().__init__(message)
        self.info = ErrorInfo(
            type=type(self).__name__,
            message=message,
            retriable=self.retriable,
            where=where,
            cause=cause,
        )


# --- Model ---------------------------------------------------------------------


class ModelError(AgentError):
    """Generic model/provider failure."""


class RateLimitError(ModelError):
    """Transient -> retry (exp+jitter, <=3)."""

    retriable = True


# --- Tools ---------------------------------------------------------------------


class ToolError(AgentError):
    """Tool logic failure -> surface to model (on_tool_error)."""


class ToolTimeout(ToolError):
    """Tool exceeded its timeout -> surface to model or raise."""


# --- Protocols (stubbed for later milestones) ----------------------------------


class MCPConnectionError(AgentError):
    """Transient -> retry/reconnect."""

    retriable = True


class A2ATaskFailed(AgentError):
    pass


# --- Memory --------------------------------------------------------------------


class MemoryError_(AgentError):
    """Non-fatal -> degrade: skip memory, continue."""


# --- Runtime / control ---------------------------------------------------------


class CheckpointError(AgentError):
    """Fatal -> terminate run (status=error)."""


class SandboxUnavailable(ToolError):
    """Fail closed when an untrusted executor is required but absent."""


class GuardrailViolation(AgentError):
    """Deterministic -> no retry; terminate run."""


class MaxStepsExceeded(AgentError):
    """Terminate; checkpoint preserved."""


class DeadlineExceeded(AgentError):
    """Terminate; checkpoint preserved."""
