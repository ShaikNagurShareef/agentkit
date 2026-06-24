"""Error taxonomy + retry disposition (§10)."""

from __future__ import annotations

from agentkit import errors


def test_error_info_carried():
    e = errors.ToolError("bad", where="mytool", cause="ValueError")
    assert e.info.type == "ToolError"
    assert e.info.message == "bad"
    assert e.info.where == "mytool"
    assert e.info.cause == "ValueError"
    assert e.info.retriable is False


def test_retriable_flags():
    assert errors.RateLimitError("x").info.retriable is True
    assert errors.MCPConnectionError("x").info.retriable is True
    assert errors.ToolError("x").info.retriable is False
    assert errors.GuardrailViolation("x").info.retriable is False


def test_hierarchy():
    assert issubclass(errors.RateLimitError, errors.ModelError)
    assert issubclass(errors.ModelError, errors.AgentError)
    assert issubclass(errors.ToolTimeout, errors.ToolError)
    assert issubclass(errors.MaxStepsExceeded, errors.AgentError)
