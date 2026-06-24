"""Multi-agent patterns: hierarchical/supervisory via Agent.as_tool()."""

from __future__ import annotations

import pytest

from agentkit import Agent
from agentkit.context import RunContext
from conftest import FakeModelProvider, tool_call


@pytest.mark.asyncio
async def test_agent_as_tool_runs_subagent():
    sub = Agent(name="specialist", model="fake:fake")
    sub._provider = FakeModelProvider(["subagent answer"])
    t = sub.as_tool()
    assert t.name == "ask_specialist" and t.source == "agent"
    res = await t.invoke({"input": "do the thing"}, RunContext(session_id="s"))
    assert res.ok and res.content == "subagent answer"


@pytest.mark.asyncio
async def test_supervisor_delegates_to_subagent():
    sub = Agent(name="researcher", model="fake:fake")
    sub._provider = FakeModelProvider(["found the facts"])

    supervisor = Agent(name="director", model="fake:fake", tools=[sub.as_tool()])
    # supervisor: call the subagent tool, then answer
    supervisor._provider = FakeModelProvider(
        [[tool_call("ask_researcher", input="research X")], "final report"]
    )
    result = await supervisor.arun("investigate X")
    assert result.status == "done"
    assert result.output == "final report"
    rec = result.tool_calls[0]
    assert rec.name == "ask_researcher"
    assert rec.content == "found the facts"   # subagent's output flowed back
