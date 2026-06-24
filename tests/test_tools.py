"""Tool decorator, schema inference, ToolResult, surface-vs-raise."""

from __future__ import annotations

import pytest

from agentkit import Tool, tool
from agentkit.context import RunContext
from agentkit.tools.base import ToolResult


def test_schema_inference_from_hints_and_docstring():
    @tool
    def add(a: int, b: int = 2) -> int:
        """Add two numbers.

        Args:
            a: the first addend
            b: the second addend
        """
        return a + b

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert "Add two numbers" in add.description
    props = add.parameters["properties"]
    assert props["a"]["type"] == "integer"
    assert props["a"]["description"] == "the first addend"
    assert "a" in add.parameters["required"]
    assert "b" not in add.parameters.get("required", [])


def test_tool_custom_name_and_timeout():
    @tool(name="search", timeout_s=5)
    def search_db(query: str) -> list[str]:
        return [query]

    assert search_db.name == "search"
    assert search_db.timeout_s == 5


@pytest.mark.asyncio
async def test_tool_invoke_success():
    @tool
    def echo(text: str) -> str:
        return text.upper()

    result = await echo.invoke({"text": "hi"}, RunContext(session_id="s"))
    assert result.ok
    assert result.content == "HI"
    assert result.latency_ms is not None


@pytest.mark.asyncio
async def test_tool_invoke_exception_becomes_tool_error():
    @tool
    def boom() -> str:
        raise ValueError("kaboom")

    result = await boom.invoke({}, RunContext(session_id="s"))
    assert not result.ok
    assert result.error is not None
    assert result.error.type == "ToolError"


@pytest.mark.asyncio
async def test_tool_ctx_injection():
    @tool
    def whoami(ctx: RunContext) -> str:
        return ctx.session_id

    result = await whoami.invoke({}, RunContext(session_id="abc"))
    assert result.content == "abc"


@pytest.mark.asyncio
async def test_tool_returning_toolresult_passthrough():
    @tool
    def custom() -> ToolResult:
        return ToolResult(ok=False, content="nope")

    result = await custom.invoke({}, RunContext(session_id="s"))
    assert not result.ok
    assert result.content == "nope"
