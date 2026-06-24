"""Runtime app: FastAPI factory, endpoints, lifespan (§3.3).

`create_app` turns an Agent *or* Flow into a local service exposing the run loop
over HTTP, with optional token streaming (SSE), durable runs, health, and metrics.
For an Agent, the graph is compiled in the lifespan against a durable async SQLite
saver so runs survive restarts and `/runs/{id}/resume` works (UC-5). A served Flow
runs the same `/invoke` surface (UC-3).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel

from ..config import db_url as default_db_url
from ..engine.compiler import GraphCompiler
from ..engine.executor import Executor
from ..types import DoneEvent, RunResult, TokenEvent
from .checkpoint import SqliteCheckpointer

if TYPE_CHECKING:
    from ..agent import Agent
    from ..flow import Flow


class InvokeRequest(BaseModel):
    input: str
    session_id: str | None = None
    stream: bool = False
    deadline_s: float | None = None


class EvalRequest(BaseModel):
    dataset: list[dict]
    metrics: list[str] = ["task_success", "tool_correctness"]
    sample: int | None = None


# --- runners: unify Agent and Flow behind one surface -------------------------


class _AgentRunner:
    def __init__(self, agent: "Agent", executor: Executor) -> None:
        self.agent = agent
        self.executor = executor
        self.sessions = agent._sessions

    async def run(self, req: InvokeRequest) -> RunResult:
        session = await self.sessions.acquire(req.session_id)
        try:
            async with self.sessions.lock(session.session_id):
                return await self.executor.arun(
                    req.input, session_id=session.session_id,
                    thread_id=session.thread_id, deadline_s=req.deadline_s,
                )
        finally:
            await self.sessions.release(session)

    async def stream(self, req: InvokeRequest):
        session = await self.sessions.acquire(req.session_id)
        try:
            async with self.sessions.lock(session.session_id):
                async for ev in self.executor.astream(
                    req.input, session_id=session.session_id,
                    thread_id=session.thread_id, deadline_s=req.deadline_s,
                ):
                    yield ev
        finally:
            await self.sessions.release(session)

    async def get_run(self, session_id: str) -> RunResult | None:
        return await self.executor.get_run(session_id=session_id, thread_id=session_id)

    async def resume(self, session_id: str) -> RunResult:
        session = await self.sessions.acquire(session_id)
        try:
            async with self.sessions.lock(session.session_id):
                return await self.executor.aresume(
                    session_id=session.session_id, thread_id=session.thread_id
                )
        finally:
            await self.sessions.release(session)


class _FlowRunner:
    def __init__(self, flow: "Flow") -> None:
        self.flow = flow

    async def run(self, req: InvokeRequest) -> RunResult:
        return await self.flow.arun(req.input, session_id=req.session_id)

    async def stream(self, req: InvokeRequest):
        # Node-level orchestration events drive the dashboard graph live.
        async for ev in self.flow.astream(req.input, session_id=req.session_id):
            yield ev

    async def get_run(self, session_id: str) -> RunResult | None:
        return None  # flows do not checkpoint in M7

    async def resume(self, session_id: str) -> RunResult:
        raise HTTPException(status_code=501, detail="resume is not supported for flows")


def _is_flow(target: Any) -> bool:
    from ..flow import Flow

    return isinstance(target, Flow)


def _sibling_db(dsn: str, suffix: str) -> str:
    """A distinct SQLite DSN so a second app doesn't lock the same file."""
    if ":memory:" in dsn or dsn in ("sqlite://", ""):
        return dsn  # in-memory connections are isolated; no shared lock
    if dsn.endswith(".db"):
        return f"{dsn[:-3]}.{suffix}.db"
    return f"{dsn}.{suffix}"


def create_app(
    target: "Agent | Flow",
    *,
    ui: str = "agent",
    dashboard_url: str | None = None,
    mcp: bool = True,
    a2a: bool = True,
    metrics: bool = True,
    auth_token: str | None = None,
    db_url: str | None = None,
    examples: list[str] | None = None,
    description: str | None = None,
) -> FastAPI:
    """Build a FastAPI app serving ``target`` (an Agent or a Flow).

    ``ui`` selects the root page: ``"agent"`` serves the end-user Agent UI (chat);
    ``"dashboard"`` serves the operator orchestrator dashboard. Both expose the
    full JSON/SSE API, so the dashboard is typically run on its own port pointing
    at the same target (and the same ``db_url`` so runs are shared).
    """
    token = auth_token or os.environ.get("AGENTKIT_AUTH_TOKEN")
    examples = examples or []
    dsn = db_url or default_db_url()
    is_flow = _is_flow(target)

    registry = CollectorRegistry()
    runs_total = Counter("agentkit_runs_total", "Total runs invoked", registry=registry)
    errors_total = Counter("agentkit_run_errors_total", "Runs ending in error", registry=registry)
    run_latency = Histogram(
        "agentkit_run_latency_seconds", "Run latency (non-streaming)", registry=registry
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if is_flow:
            app.state.runner = _FlowRunner(target)  # type: ignore[arg-type]
            app.state.ready = True
            yield
            app.state.ready = False
            return
        agent = target  # type: ignore[assignment]
        await agent._ensure_integrations()
        saver_cm = SqliteCheckpointer(dsn).saver()
        saver = await saver_cm.__aenter__()
        setup = getattr(saver, "setup", None)
        if setup is not None:
            await setup()
        graph = GraphCompiler(agent, agent.provider).compile(checkpointer=saver)
        app.state.runner = _AgentRunner(agent, Executor(agent, graph=graph))
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await saver_cm.__aexit__(None, None, None)

    title = getattr(target, "name", "agentkit")
    app = FastAPI(title=f"agentkit:{title}", lifespan=lifespan)
    app.state.ready = False

    # Protocol surfaces (§4) — agents only.
    if mcp and not is_flow:
        from ..protocols.mcp_server import MCPServer

        app.include_router(MCPServer.for_agent(target).router())  # type: ignore[arg-type]
    if a2a and not is_flow:
        from ..protocols.a2a import A2AServer

        app.include_router(A2AServer(target).router())  # type: ignore[arg-type]

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        if token is None:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.post("/invoke", dependencies=[Depends(require_auth)])
    async def invoke(req: InvokeRequest, request: Request):
        runner = request.app.state.runner
        runs_total.inc()
        if req.stream:
            async def sse():
                async for ev in runner.stream(req):
                    yield f"event: {ev.type}\ndata: {ev.model_dump_json()}\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")
        start = time.perf_counter()
        result = await runner.run(req)
        run_latency.observe(time.perf_counter() - start)
        if result.status == "error":
            errors_total.inc()
        return result

    @app.get("/runs/{session_id}", dependencies=[Depends(require_auth)])
    async def get_run(session_id: str, request: Request):
        result = await request.app.state.runner.get_run(session_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"no run for session '{session_id}'")
        return result

    @app.post("/runs/{session_id}/resume", dependencies=[Depends(require_auth)])
    async def resume_run(session_id: str, request: Request):
        return await request.app.state.runner.resume(session_id)

    # --- UI roots + introspection ---------------------------------------------

    _ui_dir = Path(__file__).parent / "dashboard"
    _no_cache = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

    def _html(name: str) -> HTMLResponse:
        return HTMLResponse((_ui_dir / name).read_text(), headers=_no_cache)

    _root_file = "index.html" if ui == "dashboard" else "agent.html"

    @app.get("/", response_class=HTMLResponse)
    async def root_ui():
        return _html(_root_file)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_ui():
        return _html("index.html")

    @app.get("/ui", response_class=HTMLResponse)
    async def agent_ui():
        return _html("agent.html")

    @app.get("/info")
    async def info():
        d = target.describe()
        tracer = getattr(target, "_tracer", None)
        obs = type(tracer).__name__.replace("Tracer", "").lower() if tracer else "none"
        return {
            "kind": d["kind"],
            "name": d["name"],
            "model": d.get("model"),
            "description": description,
            "tools": len(d.get("tools", [])),
            "mcp_servers": d.get("mcp_servers", []),
            "a2a_peers": d.get("a2a_peers", []),
            "memory": d.get("memory", False),
            "observability": obs or "noop",
            "auth": token is not None,
            "tool_names": [t.get("name") for t in d.get("tools", [])],
            "examples": examples,
            "ui": ui,
            "dashboard_url": dashboard_url,
        }

    @app.get("/graph")
    async def graph():
        return target.describe()

    @app.get("/memory")
    async def memory():
        store = getattr(target, "_memory_store", None)
        if store is None or not hasattr(store, "snapshot"):
            return {"enabled": False, "scopes": {"session": [], "user": [], "agent": []},
                    "summaries": {}, "sessions": [], "counts": {}}
        return {"enabled": True, **store.snapshot()}

    @app.get("/memory/search")
    async def memory_search(q: str, scope: str = "agent"):
        store = getattr(target, "_memory_store", None)
        if store is None:
            return {"results": []}
        from ..context import RunContext

        hits = await store.search(q, scope=scope, k=10, ctx=RunContext(session_id="dashboard"))
        return {"results": [{"text": h.text, "score": h.score, "metadata": h.metadata} for h in hits]}

    @app.post("/eval", dependencies=[Depends(require_auth)])
    async def run_eval(req: EvalRequest):
        from ..observability import (
            EvalItem,
            EvalRunner,
            Faithfulness,
            Latency,
            TaskSuccess,
            ToolCorrectness,
        )

        reg = {"task_success": TaskSuccess, "tool_correctness": ToolCorrectness,
               "faithfulness": Faithfulness, "latency": Latency}
        items = [EvalItem(**d) for d in req.dataset]
        metrics = [reg[m]() for m in req.metrics if m in reg]
        report = await EvalRunner(target, dataset=items, metrics=metrics).run(sample=req.sample)
        return report.model_dump()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz(request: Request):
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="starting")
        return {"status": "ready"}

    if metrics:
        @app.get("/metrics")
        async def metrics_endpoint():
            return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


def serve_target(
    target: "Agent | Flow",
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    dashboard_port: int | None = None,
    **kw: Any,
) -> None:
    """Serve the Agent UI on ``port`` and, if ``dashboard_port`` is set, the
    orchestrator dashboard on that separate port (sharing the same target + DB)."""
    import threading

    import uvicorn

    kw.setdefault("db_url", default_db_url())
    kw.pop("dashboard_url", None)

    if dashboard_port:
        # The dashboard runs in a second process-thread, so it needs its OWN
        # checkpoint file — two apps opening the same SQLite file and running
        # setup() concurrently raises "database is locked". The shared target
        # object keeps memory/graph/tools/eval consistent across both UIs.
        dash_kw = {**kw, "db_url": _sibling_db(kw["db_url"], "dashboard")}
        dash = create_app(target, ui="dashboard", dashboard_url=f"http://{host}:{port}", **dash_kw)
        dsrv = uvicorn.Server(uvicorn.Config(dash, host=host, port=dashboard_port, log_level="warning"))
        threading.Thread(target=dsrv.run, daemon=True).start()
        print(f"  orchestrator dashboard → http://{host}:{dashboard_port}")
        kw["dashboard_url"] = f"http://{host}:{dashboard_port}"

    app = create_app(target, ui="agent", **kw)
    print(f"  agent UI → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
