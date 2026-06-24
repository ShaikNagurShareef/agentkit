"""M7 flow engine: builder control flow, agent composition, serving, YAML."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentkit import Agent, Flow
from conftest import FakeModelProvider


@pytest.mark.asyncio
async def test_sequential_functions():
    flow = Flow("seq").step(lambda v: v.upper()).step(lambda v: v + "!")
    r = await flow.arun("hi")
    assert r.status == "done"
    assert r.output == "HI!"


@pytest.mark.asyncio
async def test_parallel_fanout():
    flow = Flow("par").parallel(lambda v: v + "a", lambda v: v + "b")
    r = await flow.arun("x")
    assert "xa" in r.output and "xb" in r.output


@pytest.mark.asyncio
async def test_conditional():
    flow = Flow("c").when(lambda v: "urgent" in v).then(lambda v: "ESCALATE").otherwise(lambda v: "normal")
    assert (await flow.arun("urgent ticket")).output == "ESCALATE"
    assert (await flow.arun("hello")).output == "normal"


@pytest.mark.asyncio
async def test_loop_until():
    flow = Flow("l").loop(lambda v: v + 1, until=lambda v: v >= 3, max_iter=10)
    assert (await flow.arun(0)).output == "3"


@pytest.mark.asyncio
async def test_map_over_list():
    flow = Flow("m").map(lambda v: v * 2)
    out = (await flow.arun([1, 2, 3])).output
    assert "2" in out and "4" in out and "6" in out


@pytest.mark.asyncio
async def test_agent_composition():
    a1 = Agent(name="researcher", model="fake:fake")
    a1._provider = FakeModelProvider(["researched the topic"])
    a2 = Agent(name="writer", model="fake:fake")
    a2._provider = FakeModelProvider(["final article"])
    flow = Flow("research-and-write").step(a1).step(a2)
    r = await flow.arun("quantum computing")
    assert r.output == "final article"
    assert r.usage.total_tokens > 0  # usage accumulated across agents


def test_sync_run():
    flow = Flow("s").step(lambda v: v[::-1])
    assert flow.run("abc").output == "cba"


def test_flow_served_over_http(tmp_path):
    flow = Flow("served").step(lambda v: v.upper())
    app = flow.asgi_app(db_url=f"sqlite:///{tmp_path/'f.db'}")
    with TestClient(app) as client:
        r = client.post("/invoke", json={"input": "hello"})
        assert r.status_code == 200
        assert r.json()["output"] == "HELLO"
        assert client.get("/health").status_code == 200


def test_flow_from_yaml(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    # two agent specs + a flow spec wiring them sequentially
    (tmp_path / "researcher.yaml").write_text(
        yaml.safe_dump({"apiVersion": "agentkit/v1", "kind": "Agent",
                        "metadata": {"name": "researcher"}, "spec": {"model": "anthropic:claude-opus-4-8"}})
    )
    (tmp_path / "writer.yaml").write_text(
        yaml.safe_dump({"apiVersion": "agentkit/v1", "kind": "Agent",
                        "metadata": {"name": "writer"}, "spec": {"model": "anthropic:claude-opus-4-8"}})
    )
    flow_yaml = {
        "apiVersion": "agentkit/v1", "kind": "Flow", "metadata": {"name": "rw"},
        "spec": {
            "entry": "researcher", "finish": "writer",
            "nodes": [
                {"id": "researcher", "kind": "agent", "ref": str(tmp_path / "researcher.yaml")},
                {"id": "writer", "kind": "agent", "ref": str(tmp_path / "writer.yaml")},
            ],
            "edges": [{"src": "researcher", "dst": "writer"}],
        },
    }
    (tmp_path / "flow.yaml").write_text(yaml.safe_dump(flow_yaml))

    flow = Flow.from_yaml(str(tmp_path / "flow.yaml"))
    assert flow.name == "rw"
    assert len(flow._steps) == 2  # researcher -> writer


def test_load_target_agent_yaml(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    from agentkit.spec import load_target

    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump({"apiVersion": "agentkit/v1", "kind": "Agent",
                        "metadata": {"name": "solo"},
                        "spec": {"model": "anthropic:claude-opus-4-8", "instructions": "help"}})
    )
    agent = load_target(str(tmp_path / "a.yaml"))
    assert isinstance(agent, Agent)
    assert agent.name == "solo" and agent.model == "anthropic:claude-opus-4-8"
