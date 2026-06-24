"""HTTP serving tests (M2) — ASGI TestClient + FakeModelProvider, no API key."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentkit import Agent, tool
from conftest import FakeModelProvider, tool_call


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _app(script, *, db, tools=None, **kw):
    agent = Agent(name="srv", model="fake:fake", tools=tools or [], **kw)
    agent._provider = FakeModelProvider(script)
    return agent.asgi_app(db_url=f"sqlite:///{db}")


def test_invoke_nonstream_direct_answer(tmp_path):
    app = _app(["hello over http"], db=tmp_path / "a.db")
    with TestClient(app) as client:
        r = client.post("/invoke", json={"input": "hi"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "done"
        assert body["output"] == "hello over http"


def test_invoke_tool_loop(tmp_path):
    app = _app(
        [[tool_call("add", a=2, b=3)], "the sum is 5"], db=tmp_path / "b.db", tools=[add]
    )
    with TestClient(app) as client:
        r = client.post("/invoke", json={"input": "2+3?", "session_id": "s-tool"})
        body = r.json()
        assert body["status"] == "done"
        assert body["output"] == "the sum is 5"
        assert body["tool_calls"][0]["name"] == "add"
        assert body["tool_calls"][0]["content"] == 5


def test_invoke_streaming_sse(tmp_path):
    app = _app(
        [[tool_call("add", a=1, b=2)], "streamed answer"],
        db=tmp_path / "c.db",
        tools=[add],
    )
    with TestClient(app) as client:
        r = client.post("/invoke", json={"input": "go", "stream": True})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = r.text
        assert "event: tool_start" in text
        assert "event: tool_end" in text
        assert "event: token" in text  # final text step streams a token chunk
        assert "event: done" in text


def test_get_run_and_404(tmp_path):
    app = _app(["answer"], db=tmp_path / "d.db")
    with TestClient(app) as client:
        client.post("/invoke", json={"input": "hi", "session_id": "known"})
        r = client.get("/runs/known")
        assert r.status_code == 200
        assert r.json()["output"] == "answer"
        assert client.get("/runs/missing").status_code == 404


def test_resume(tmp_path):
    app = _app(["done answer"], db=tmp_path / "e.db")
    with TestClient(app) as client:
        client.post("/invoke", json={"input": "hi", "session_id": "r1"})
        r = client.post("/runs/r1/resume")
        assert r.status_code == 200
        assert r.json()["status"] in ("done", "interrupted")


def test_health_and_metrics(tmp_path):
    app = _app(["x"], db=tmp_path / "f.db")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/healthz").json()["status"] == "ready"
        m = client.get("/metrics")
        assert m.status_code == 200
        assert "agentkit_runs_total" in m.text


def test_auth_required_when_token_set(tmp_path):
    agent = Agent(name="srv", model="fake:fake")
    agent._provider = FakeModelProvider(["secret answer"])
    app = agent.asgi_app(db_url=f"sqlite:///{tmp_path / 'g.db'}", auth_token="s3cr3t")
    with TestClient(app) as client:
        assert client.post("/invoke", json={"input": "hi"}).status_code == 401
        ok = client.post(
            "/invoke",
            json={"input": "hi"},
            headers={"Authorization": "Bearer s3cr3t"},
        )
        assert ok.status_code == 200
        # health is unauthenticated
        assert client.get("/health").status_code == 200


def test_durability_across_app_restart(tmp_path):
    db = tmp_path / "shared.db"
    # First "process": run and persist.
    app1 = _app(["persisted answer"], db=db)
    with TestClient(app1) as c1:
        c1.post("/invoke", json={"input": "hi", "session_id": "dur1"})

    # Second "process": a fresh agent/app over the same DB still sees the run.
    app2 = _app(["unused"], db=db)
    with TestClient(app2) as c2:
        r = c2.get("/runs/dur1")
        assert r.status_code == 200
        assert r.json()["output"] == "persisted answer"
