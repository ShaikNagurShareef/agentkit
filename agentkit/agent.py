"""Agent core — the public SDK surface (§3.1.1).

An Agent is a configured reasoning unit (model + tools + memory + protocol
bindings). It compiles to a four-node graph and runs in-process via run/arun.
Serving (`serve`/`asgi_app`) lands in M2.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .context import RunContext
from .engine.compiler import CompiledGraph, GraphCompiler
from .engine.executor import Executor
from .models.base import ModelProvider, ModelSettings, resolve_provider
from .runtime.checkpoint import default_saver
from .runtime.session import SessionManager
from .tools.base import Tool, ToolResult, tool as _tool
from .types import Message, RunEvent, RunResult


class Agent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    model: str
    instructions: str | None = None
    tools: list[Any] = Field(default_factory=list)  # Tool | Callable
    mcp_servers: list[str] = Field(default_factory=list)  # external MCP server URLs (FR-5)
    a2a_peers: list[str] = Field(default_factory=list)  # A2A peer agent-card URLs (FR-7)
    memory: Any = None  # MemoryConfig | str | None
    observability: Any = "langfuse"  # ObsConfig | str | None
    max_steps: int = 20
    step_timeout_s: float = 120
    on_tool_error: Literal["surface", "raise"] = "surface"
    model_settings: ModelSettings = Field(default_factory=ModelSettings)

    _resolved_tools: list[Tool] = PrivateAttr(default_factory=list)
    _provider: ModelProvider | None = PrivateAttr(default=None)
    _graph: CompiledGraph | None = PrivateAttr(default=None)
    _checkpointer: Any = PrivateAttr(default=None)
    _sessions: SessionManager | None = PrivateAttr(default=None)
    _memory_store: Any = PrivateAttr(default=None)
    _memory_cfg: Any = PrivateAttr(default=None)
    _tracer: Any = PrivateAttr(default=None)
    _integrations_connected: bool = PrivateAttr(default=False)
    _mcp_clients: list = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        # Normalize tools: wrap bare callables, accept Tool instances directly.
        self._resolved_tools = [self._coerce_tool(t) for t in self.tools]
        self._checkpointer = default_saver()
        self._sessions = SessionManager()
        self._memory_store = self._build_memory_store()
        self._tracer = self._build_tracer()

    def _build_memory_store(self) -> Any:
        """Build a memory store if memory is configured (else None)."""
        if self.memory is None:
            return None
        from .memory import MemoryConfig, build_memory_store

        if isinstance(self.memory, MemoryConfig):
            cfg = self.memory
        elif isinstance(self.memory, str):
            # shorthand like "local" or "mem0://local"
            backend = self.memory.split("://", 1)[-1].split("/", 1)[0] or "local"
            cfg = MemoryConfig(backend=backend if backend in (
                "local", "sqlite_vec", "chroma", "qdrant", "pgvector") else "local")
        elif isinstance(self.memory, dict):
            cfg = MemoryConfig(**self.memory)
        else:
            cfg = MemoryConfig()
        self._memory_cfg = cfg
        return build_memory_store(cfg)

    def _build_tracer(self) -> Any:
        """Build the observability tracer (no-op unless configured)."""
        from .observability import build_tracer

        return build_tracer(self.observability)

    # --- tools -----------------------------------------------------------------

    @staticmethod
    def _coerce_tool(t: Tool | Callable) -> Tool:
        if isinstance(t, Tool):
            return t
        coerced = _tool(t)
        assert isinstance(coerced, Tool)
        return coerced

    @property
    def resolved_tools(self) -> list[Tool]:
        return self._resolved_tools

    def register_tool(self, fn: Callable) -> Tool:
        """Wrap and register a tool, invalidating the cached graph."""
        t = self._coerce_tool(fn)
        self._resolved_tools.append(t)
        self._graph = None
        return t

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> Tool:
        """Expose this agent as a Tool so a supervisor can delegate to it.

        Enables the hierarchical / supervisory pattern: a supervisor
        ``Agent(tools=[specialist.as_tool(), ...])`` calls subagents through its
        ReAct loop. The subagent runs as a full agent (its own tools/memory) and
        returns its answer.
        """

        async def handler(args: dict, ctx: RunContext) -> ToolResult:
            result = await self._arun_result(
                args.get("input", ""), session_id=None, deadline_s=None
            )
            return ToolResult(
                ok=result.status == "done",
                content=result.output,
                raw={"tool_calls": [r.model_dump() for r in result.tool_calls],
                     "usage": result.usage.model_dump()},
            )

        return Tool(
            name=name or f"ask_{self.name.replace('-', '_')}",
            description=description or f"Delegate a sub-task to the '{self.name}' specialist agent.",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string", "description": "the sub-task for the agent"}},
                "required": ["input"],
            },
            source="agent",
            timeout_s=300,
            handler=handler,
        )

    async def _ensure_integrations(self) -> None:
        """Connect external MCP servers and A2A peers, surfacing them as tools.

        Idempotent; failures degrade gracefully (logged, not fatal) so a flaky
        peer never breaks a run.
        """
        if self._integrations_connected:
            return
        self._integrations_connected = True
        if not (self.mcp_servers or self.a2a_peers):
            return

        import logging

        from .protocols.a2a import peer_as_tool
        from .protocols.mcp_client import MCPClient

        log = logging.getLogger("agentkit.integrations")
        added = False
        for i, url in enumerate(self.mcp_servers):
            try:
                client = MCPClient(url, namespace=f"mcp{i}")
                await client.connect()
                self._resolved_tools.extend(client.as_tools())
                self._mcp_clients.append(client)
                added = True
            except Exception as e:  # noqa: BLE001
                log.warning("MCP server %s unavailable: %s", url, e)
        for peer in self.a2a_peers:
            try:
                self._resolved_tools.append(await peer_as_tool(peer))
                added = True
            except Exception as e:  # noqa: BLE001
                log.warning("A2A peer %s unavailable: %s", peer, e)
        if added:
            self._graph = None  # rebuild graph with the new tools

    # --- model + graph ---------------------------------------------------------

    @property
    def provider(self) -> ModelProvider:
        if self._provider is None:
            self._provider = resolve_provider(self.model, self.model_settings)
        return self._provider

    def describe(self) -> dict:
        """Topology + capability description for the dashboard (§3.1.2)."""
        nodes = [
            {"id": "__start__", "label": "start", "kind": "start"},
            {"id": "memory_read", "label": "memory_read", "kind": "node"},
            {"id": "model", "label": "model", "kind": "model"},
            {"id": "tool", "label": "tool", "kind": "tool"},
            {"id": "memory_write", "label": "memory_write", "kind": "node"},
            {"id": "__end__", "label": "end", "kind": "end"},
        ]
        edges = [
            {"src": "__start__", "dst": "memory_read"},
            {"src": "memory_read", "dst": "model"},
            {"src": "model", "dst": "tool", "label": "tool_calls"},
            {"src": "tool", "dst": "model", "label": "results"},
            {"src": "model", "dst": "memory_write", "label": "final"},
            {"src": "memory_write", "dst": "__end__"},
        ]
        return {
            "kind": "agent",
            "name": self.name,
            "model": self.model,
            "instructions": self.instructions,
            "nodes": nodes,
            "edges": edges,
            "tools": [
                {"name": t.name, "description": t.description, "source": t.source}
                for t in self._resolved_tools
            ],
            "mcp_servers": list(self.mcp_servers),
            "a2a_peers": list(self.a2a_peers),
            "memory": self.memory is not None,
        }

    def as_graph(self) -> CompiledGraph:
        """Compile (and cache) the agent's graph."""
        if self._graph is None:
            compiler = GraphCompiler(self, self.provider)
            self._graph = compiler.compile(checkpointer=self._checkpointer)
        return self._graph

    # --- execution -------------------------------------------------------------

    def run(
        self,
        input: str | Message,
        *,
        session_id: str | None = None,
        stream: bool = False,
        deadline_s: float | None = None,
    ) -> RunResult:
        """Synchronous run (wraps arun via asyncio.run, §2.3).

        Streaming is async-only; ``stream=True`` here runs to completion and
        returns the final RunResult.
        """
        return asyncio.run(
            self._arun_result(input, session_id=session_id, deadline_s=deadline_s)
        )

    async def arun(
        self,
        input: str | Message,
        *,
        session_id: str | None = None,
        stream: bool = False,
        deadline_s: float | None = None,
    ) -> RunResult | AsyncIterator[RunEvent]:
        """Asynchronous run. With ``stream=True`` returns an async iterator of RunEvents."""
        if stream:
            return self._arun_stream(input, session_id=session_id, deadline_s=deadline_s)
        return await self._arun_result(
            input, session_id=session_id, deadline_s=deadline_s
        )

    async def _arun_result(
        self,
        input: str | Message,
        *,
        session_id: str | None,
        deadline_s: float | None,
    ) -> RunResult:
        assert self._sessions is not None
        await self._ensure_integrations()
        session = await self._sessions.acquire(session_id)
        try:
            async with self._sessions.lock(session.session_id):
                return await Executor(self).arun(
                    input,
                    session_id=session.session_id,
                    thread_id=session.thread_id,
                    deadline_s=deadline_s,
                )
        finally:
            await self._sessions.release(session)

    async def _arun_stream(
        self,
        input: str | Message,
        *,
        session_id: str | None,
        deadline_s: float | None,
    ) -> AsyncIterator[RunEvent]:
        assert self._sessions is not None
        await self._ensure_integrations()
        session = await self._sessions.acquire(session_id)
        try:
            async with self._sessions.lock(session.session_id):
                async for ev in Executor(self).astream(
                    input,
                    session_id=session.session_id,
                    thread_id=session.thread_id,
                    deadline_s=deadline_s,
                ):
                    yield ev
        finally:
            await self._sessions.release(session)

    # --- serving (M2) ----------------------------------------------------------

    def asgi_app(self, **kw: Any) -> Any:
        """Return a FastAPI app + dashboard serving this agent (for uvicorn/embedding)."""
        from .runtime.app import create_app

        return create_app(self, **kw)

    def serve(
        self, host: str = "127.0.0.1", port: int = 8080, *, dashboard_port: int | None = None, **kw: Any
    ) -> None:
        """Serve the Agent UI (chat) on ``port`` and, if ``dashboard_port`` is set,
        the separate orchestrator dashboard on that port (§11; binds 127.0.0.1).

        Accepts create_app kwargs: ``mcp``, ``a2a``, ``metrics``, ``auth_token``,
        ``db_url``, ``examples``, ``description``.
        """
        from .runtime.app import serve_target

        serve_target(self, host=host, port=port, dashboard_port=dashboard_port, **kw)
