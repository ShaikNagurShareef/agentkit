"""End-to-end engine loop with a scripted model (no API key)."""

from __future__ import annotations

import pytest

from agentkit import tool
from agentkit.types import RunResult
from conftest import tool_call


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@pytest.mark.asyncio
async def test_tool_using_loop(make_agent):
    # model: request add(2,3) -> sees result -> answers "5"
    agent = make_agent([[tool_call("add", a=2, b=3)], "The answer is 5"], tools=[add])
    result = await agent.arun("what is 2+3?")
    assert isinstance(result, RunResult)
    assert result.status == "done"
    assert result.output == "The answer is 5"
    # the tool ran and was recorded
    assert len(result.tool_calls) == 1
    rec = result.tool_calls[0]
    assert rec.name == "add" and rec.ok and rec.content == 5
    # usage accumulated across the two model calls
    assert result.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_no_tool_direct_answer(make_agent):
    agent = make_agent(["hello there"])
    result = await agent.arun("hi")
    assert result.status == "done"
    assert result.output == "hello there"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_max_steps_exceeded(make_agent):
    # Always request a tool -> never terminates -> hits max_steps.
    agent = make_agent(
        [[tool_call("add", a=1, b=1)]] * 10, tools=[add], max_steps=3
    )
    result = await agent.arun("loop")
    assert result.status == "error"
    assert result.error is not None
    assert result.error.type == "MaxStepsExceeded"


@pytest.mark.asyncio
async def test_deadline_exceeded(make_agent):
    agent = make_agent(["never reached"])
    result = await agent.arun("hi", deadline_s=-1)  # already expired
    assert result.status == "error"
    assert result.error.type == "DeadlineExceeded"


@pytest.mark.asyncio
async def test_on_tool_error_surface(make_agent):
    @tool
    def boom() -> str:
        raise ValueError("kaboom")

    # surface (default): tool failure is fed back, model then answers.
    agent = make_agent(
        [[tool_call("boom")], "recovered"], tools=[boom], on_tool_error="surface"
    )
    result = await agent.arun("go")
    assert result.status == "done"
    assert result.output == "recovered"
    assert result.tool_calls[0].ok is False


@pytest.mark.asyncio
async def test_on_tool_error_raise(make_agent):
    @tool
    def boom() -> str:
        raise ValueError("kaboom")

    agent = make_agent([[tool_call("boom")]], tools=[boom], on_tool_error="raise")
    result = await agent.arun("go")
    assert result.status == "error"
    assert result.error.type == "ToolError"


@pytest.mark.asyncio
async def test_streaming_events(make_agent):
    agent = make_agent([[tool_call("add", a=1, b=2)], "done: 3"], tools=[add])
    types_seen = []
    async for ev in await agent.arun("go", stream=True):
        types_seen.append(ev.type)
    assert "step" in types_seen
    assert "tool_start" in types_seen
    assert "tool_end" in types_seen
    assert types_seen[-1] == "done"


@pytest.mark.asyncio
async def test_sync_run(make_agent):
    agent = make_agent(["sync answer"])
    # run() uses asyncio.run internally; call from a thread to avoid loop conflicts.
    import asyncio

    result = await asyncio.to_thread(agent.run, "hi")
    assert result.output == "sync answer"
