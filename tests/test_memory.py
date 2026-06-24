"""M4 memory: store unit behavior + end-to-end read/write injection."""

from __future__ import annotations

import pytest

from agentkit import Agent
from agentkit.context import RunContext
from agentkit.memory import LocalMemoryStore, MemoryConfig
from agentkit.models.base import ModelResponse, ModelSettings
from agentkit.types import Message, Usage


@pytest.mark.asyncio
async def test_store_add_search_forget():
    store = LocalMemoryStore()
    ctx = RunContext(session_id="s1")
    await store.add(["I prefer Python", "the sky is blue"], scope="agent", ctx=ctx)
    hits = await store.search("which language do I like?", scope="agent", k=2, ctx=ctx)
    assert hits and hits[0].text == "I prefer Python"
    removed = await store.forget(scope="agent", filter={"session_id": "s1"})
    assert removed == 2
    assert await store.search("python", scope="agent", ctx=ctx) == []


class CapturingProvider:
    """Records the instructions passed on each model call; returns a fixed answer."""

    model = "fake"

    def __init__(self):
        self.instructions_seen: list[str | None] = []

    async def complete(self, messages, *, tools, settings, ctx, instructions=None):
        self.instructions_seen.append(instructions)
        return ModelResponse(
            message=Message(role="assistant", content="noted"),
            tool_calls=[],
            usage=Usage(total_tokens=1),
        )

    async def stream(self, *a, **k):  # pragma: no cover - not used here
        yield None


@pytest.mark.asyncio
async def test_memory_write_then_read_injection():
    agent = Agent(
        name="m",
        model="fake:fake",
        memory=MemoryConfig(strategies=["semantic", "user_preference", "summary"]),
    )
    provider = CapturingProvider()
    agent._provider = provider

    # First run states a preference -> extraction populates long-term memory.
    await agent.arun("I prefer Python and I use tabs", session_id="u1")
    assert agent._memory_store is not None
    # Second run -> memory_read recalls and injects into the model's instructions.
    await agent.arun("what language do I prefer?", session_id="u1")

    injected = provider.instructions_seen[-1] or ""
    assert "Relevant memory" in injected
    assert "Python" in injected


@pytest.mark.asyncio
async def test_no_memory_by_default():
    agent = Agent(name="n", model="fake:fake")
    assert agent._memory_store is None


@pytest.mark.asyncio
async def test_memory_captured_at_all_scopes():
    # A run should populate session (short-term), user (per-user), and agent (global).
    agent = Agent(
        name="m", model="fake:fake",
        memory=MemoryConfig(strategies=["semantic", "user_preference", "summary"]),
    )
    agent._provider = CapturingProvider()
    await agent.arun("I prefer Python and I use tabs", session_id="user-7")
    snap = agent._memory_store.snapshot()
    assert snap["counts"]["session"] > 0, "session (short-term) scope empty"
    assert snap["counts"]["user"] > 0, "user (per-user long-term) scope empty"
    assert snap["counts"]["agent"] > 0, "agent (global long-term) scope empty"
    assert "user-7" in snap["sessions"]  # transcript recorded
