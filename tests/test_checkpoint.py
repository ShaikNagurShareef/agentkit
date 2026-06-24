"""Checkpointer: in-process session continuity + sqlite DSN parsing."""

from __future__ import annotations

import pytest

from agentkit.runtime.checkpoint import SqliteCheckpointer, _sqlite_path_from_dsn


@pytest.mark.asyncio
async def test_session_continuity_across_runs(make_agent):
    # Same session_id reuses the thread; the second run's transcript includes
    # the first turn's messages (checkpoint replay via the messages reducer).
    agent = make_agent(["first answer", "second answer"])
    r1 = await agent.arun("hello", session_id="sess-A")
    assert r1.output == "first answer"
    n1 = len(r1.messages)

    r2 = await agent.arun("again", session_id="sess-A")
    assert r2.output == "second answer"
    assert len(r2.messages) > n1  # prior turn was carried forward


@pytest.mark.asyncio
async def test_separate_sessions_isolated(make_agent):
    agent = make_agent(["a1", "b1"])
    ra = await agent.arun("x", session_id="A")
    rb = await agent.arun("y", session_id="B")
    # B is a fresh thread, not contaminated by A's history.
    assert all(m.content != "x" for m in rb.messages if m.role == "user")


def test_sqlite_dsn_parsing():
    assert _sqlite_path_from_dsn("sqlite:///./agentkit.db") == "./agentkit.db"
    assert _sqlite_path_from_dsn("sqlite://") == ":memory:"
    cp = SqliteCheckpointer("sqlite:///data/x.db")
    assert cp.path == "data/x.db"
