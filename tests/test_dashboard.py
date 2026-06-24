"""Dashboard + introspection endpoints: /, /info, /graph, /memory, /eval, node stream."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentkit import Agent, Flow, tool
from agentkit.memory import MemoryConfig
from conftest import FakeModelProvider


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _agent_app(script, tmp_path, *, ui="agent", agent_kw=None, **kw):
    agent = Agent(name="dash", model="fake:fake", tools=[add], **(agent_kw or {}))
    agent._provider = FakeModelProvider(script)
    return agent.asgi_app(ui=ui, db_url=f"sqlite:///{tmp_path/'d.db'}",
                          examples=["hi there"], description="demo agent", **kw)


def test_agent_ui_and_dashboard_split(tmp_path):
    # Default root = Agent UI (chat); dashboard reachable at /dashboard.
    app = _agent_app(["ok"], tmp_path, dashboard_url="http://127.0.0.1:8911")
    with TestClient(app) as c:
        root = c.get("/").text
        assert "message the agent" in root.lower()              # Agent UI
        assert "orchestrator" in c.get("/dashboard").text.lower()  # dashboard
        info = c.get("/info").json()
        assert info["ui"] == "agent"
        assert info["dashboard_url"] == "http://127.0.0.1:8911"
    # ui="dashboard" serves the dashboard at root
    dapp = _agent_app(["ok"], tmp_path, ui="dashboard")
    with TestClient(dapp) as c:
        assert "orchestrator" in c.get("/").text.lower()
        assert c.get("/info").json()["ui"] == "dashboard"


def test_info_and_graph(tmp_path):
    app = _agent_app(["ok"], tmp_path, agent_kw={"memory": MemoryConfig()})
    with TestClient(app) as c:
        info = c.get("/info").json()
        assert info["kind"] == "agent" and info["tools"] == 1
        assert info["memory"] is True and "add" in info["tool_names"]
        assert info["examples"] == ["hi there"]
        g = c.get("/graph").json()
        assert {n["id"] for n in g["nodes"]} >= {"model", "tool", "memory_read"}


def test_memory_endpoint(tmp_path):
    app = _agent_app(["noted"], tmp_path, agent_kw={"memory": MemoryConfig()})
    with TestClient(app) as c:
        c.post("/invoke", json={"input": "I prefer Python", "session_id": "u"})
        m = c.get("/memory").json()
        assert m["enabled"] is True
        assert sum(m["counts"].values()) > 0
        # no-memory agent reports disabled
    app2 = _agent_app(["x"], tmp_path)
    with TestClient(app2) as c2:
        assert c2.get("/memory").json()["enabled"] is False


def test_eval_endpoint(tmp_path):
    app = _agent_app(["the answer is 5", "the answer is 5"], tmp_path)
    with TestClient(app) as c:
        r = c.post("/eval", json={"dataset": [{"input": "q1", "expected": "5"}],
                                  "metrics": ["task_success"]})
        assert r.status_code == 200
        assert r.json()["metrics"]["task_success"] == 1.0


def test_flow_graph_and_node_stream(tmp_path):
    flow = (Flow("demo").step(lambda v: v.upper())
            .parallel(lambda v: v + "!", lambda v: v + "?"))
    app = flow.asgi_app(db_url=f"sqlite:///{tmp_path/'f.db'}")
    with TestClient(app) as c:
        g = c.get("/graph").json()
        assert g["kind"] == "flow"
        assert any(n["kind"] == "merge" for n in g["nodes"])  # parallel fan-in
        body = c.post("/invoke", json={"input": "hi", "stream": True}).text
        assert "event: node_start" in body and "event: node_end" in body
        assert "event: done" in body
