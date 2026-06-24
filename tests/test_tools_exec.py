"""M6 tools: code execution (real, offline) + trust policy + optional-dep gates."""

from __future__ import annotations

import pytest

from agentkit.context import RunContext
from agentkit.errors import SandboxUnavailable, ToolError
from agentkit.tools.browser import BrowserTool
from agentkit.tools.code import SubprocessExecutor, build_executor, code_tool
from agentkit.tools.computer import computer_tool


@pytest.mark.asyncio
async def test_subprocess_executor_runs_python():
    ex = SubprocessExecutor()
    result = await ex.run("print('hello from subprocess')")
    assert result.exit_code == 0
    assert "hello from subprocess" in result.stdout
    assert not result.timed_out


@pytest.mark.asyncio
async def test_subprocess_executor_timeout():
    ex = SubprocessExecutor()
    result = await ex.run("import time; time.sleep(5)", timeout_s=0.3)
    assert result.timed_out


@pytest.mark.asyncio
async def test_subprocess_executor_artifacts():
    ex = SubprocessExecutor()
    result = await ex.run("open('out.txt','w').write('data')")
    assert result.artifacts.get("out.txt") == b"data"


@pytest.mark.asyncio
async def test_code_tool_end_to_end():
    t = code_tool(trust="trusted")
    res = await t.invoke({"code": "print(2 + 2)"}, RunContext(session_id="s"))
    assert res.ok
    assert res.content.strip() == "4"


def test_untrusted_without_e2b_fails_closed(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(SandboxUnavailable):
        build_executor("untrusted")


@pytest.mark.asyncio
async def test_browser_requires_optional_dep():
    # browser-use is not installed in the test env -> clear ToolError, not a crash.
    with pytest.raises(ToolError):
        await BrowserTool().run_task("go to example.com", ctx=RunContext(session_id="s"))


def test_computer_disabled_by_default():
    with pytest.raises(ToolError):
        computer_tool(enable_computer_use=False)
