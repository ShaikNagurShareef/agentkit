"""M3 protocols: MCP server/client + A2A server (no network; ASGI transport)."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from agentkit import Agent, tool
from agentkit.context import RunContext
from agentkit.protocols.a2a import A2AClient, artifact_text
from agentkit.protocols.mcp_client import MCPClient
from conftest import FakeModelProvider


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _agent(script=None, **kw):
    agent = Agent(name="proto", model="fake:fake", tools=[add], **kw)
    agent._provider = FakeModelProvider(script or ["ok"])
    return agent


def test_mcp_server_jsonrpc(tmp_path):
    app = _agent(["agent answer"]).asgi_app(db_url=f"sqlite:///{tmp_path/'m.db'}", a2a=False)
    client = TestClient(app)  # no lifespan needed for /mcp tool calls
    # tools/list exposes the local tool + invoke_agent meta-tool
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in r.json()["result"]["tools"]}
    assert "add" in names and "invoke_agent" in names
    # tools/call runs the tool
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "add", "arguments": {"a": 2, "b": 3}}},
    )
    res = r.json()["result"]
    assert res["isError"] is False
    assert res["content"][0]["text"] == "5"


def test_mcp_server_invoke_agent_meta_tool(tmp_path):
    app = _agent(["meta answer"]).asgi_app(db_url=f"sqlite:///{tmp_path/'m2.db'}", a2a=False)
    with TestClient(app) as client:
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "invoke_agent", "arguments": {"input": "hi"}}},
        )
        assert r.json()["result"]["content"][0]["text"] == "meta answer"


@pytest.mark.asyncio
async def test_mcp_client_consumes_server(tmp_path):
    app = _agent().asgi_app(db_url=f"sqlite:///{tmp_path/'m3.db'}", a2a=False)
    mc = MCPClient("http://testserver/mcp", namespace="ext")
    mc._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )
    await mc.connect()
    descs = {d.name for d in await mc.list_tools()}
    assert "add" in descs
    tools = {t.name: t for t in mc.as_tools()}
    assert "ext.add" in tools
    result = await tools["ext.add"].invoke({"a": 4, "b": 5}, RunContext(session_id="s"))
    assert result.ok and result.content == 9
    await mc.close()


def test_a2a_card_and_task(tmp_path):
    app = _agent(["a2a result"]).asgi_app(db_url=f"sqlite:///{tmp_path/'a.db'}", mcp=False)
    with TestClient(app) as client:
        card = client.get("/.well-known/agent.json").json()
        assert card["name"] == "proto"
        r = client.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": 1, "method": "tasks/send",
                  "params": {"message": {"role": "user", "parts": [{"type": "text", "text": "do it"}]}}},
        )
        task = r.json()["result"]
        assert task["status"]["state"] == "completed"
        assert artifact_text(task) == "a2a result"
        # tasks/get round-trips
        got = client.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": task["id"]}},
        )
        assert got.json()["result"]["id"] == task["id"]


def test_a2a_client_url_helper():
    assert A2AClient._task_url("http://x:8081") == "http://x:8081/a2a"
    assert A2AClient._task_url("http://x:8081/a2a") == "http://x:8081/a2a"
    assert A2AClient._task_url("http://x:8081/.well-known/agent.json") == "http://x:8081/a2a"
