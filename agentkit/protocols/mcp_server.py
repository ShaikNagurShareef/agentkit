"""MCP server (§4.2): expose the agent's tools (and an invoke_agent meta-tool).

Implements MCP's JSON-RPC method surface over Streamable HTTP (`tools/list`,
`tools/call`, `initialize`). When an Agent is served with ``mcp=True`` the runtime
mounts an auto-generated server at ``/mcp`` exposing the agent's local tools plus
an ``invoke_agent`` meta-tool that runs the agent itself as a single tool.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, Request

from ..context import RunContext
from ..tools.base import Tool, tool as _tool

if TYPE_CHECKING:
    from ..agent import Agent

PROTOCOL_VERSION = "2025-06-18"


class MCPServer:
    """Serves a set of Tools over the MCP JSON-RPC surface."""

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, Tool] = {}

    # --- registration ----------------------------------------------------------

    def register(self, t: Tool) -> Tool:
        self._tools[t.name] = t
        return t

    def tool(self, fn: Callable | None = None, *, name: str | None = None, timeout_s: float = 30):
        """Decorator: register a function as an MCP-exposed tool."""

        def wrap(func: Callable) -> Tool:
            t = _tool(func, name=name, timeout_s=timeout_s)
            assert isinstance(t, Tool)
            return self.register(t)

        return wrap(fn) if fn is not None else wrap

    @classmethod
    def for_agent(cls, agent: "Agent") -> "MCPServer":
        """Build a server exposing an agent's local tools + invoke_agent."""
        server = cls(name=agent.name, version="0.1.0")
        for t in agent.resolved_tools:
            if t.source == "local":
                server.register(t)

        @server.tool(name="invoke_agent", timeout_s=600)
        async def invoke_agent(input: str) -> str:
            """Run the agent end-to-end on a single input and return its answer."""
            result = await agent._arun_result(input, session_id=None, deadline_s=None)
            return result.output

        return server

    # --- JSON-RPC dispatch ------------------------------------------------------

    def _descriptor(self, t: Tool) -> dict:
        return {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.parameters or {"type": "object", "properties": {}},
        }

    async def handle(self, payload: dict, ctx: RunContext | None = None) -> dict:
        """Dispatch a single JSON-RPC request object and return the response."""
        rid = payload.get("id")
        method = payload.get("method")
        params = payload.get("params") or {}
        try:
            result = await self._dispatch(method, params, ctx)
        except _RpcError as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": e.code, "message": e.message}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    async def _dispatch(self, method: str, params: dict, ctx: RunContext | None) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": self.name, "version": self.version},
                "capabilities": {"tools": {"listChanged": False}},
            }
        if method in ("tools/list", "list_tools"):
            return {"tools": [self._descriptor(t) for t in self._tools.values()]}
        if method in ("tools/call", "call_tool"):
            name = params.get("name")
            args = params.get("arguments") or {}
            t = self._tools.get(name)
            if t is None:
                raise _RpcError(-32602, f"unknown tool '{name}'")
            result = await t.invoke(args, ctx or RunContext(session_id="mcp"))
            text = result.content if isinstance(result.content, str) else json.dumps(result.content)
            return {
                "content": [{"type": "text", "text": text if result.ok else (
                    result.error.message if result.error else "tool failed")}],
                "isError": not result.ok,
            }
        raise _RpcError(-32601, f"method not found: {method}")

    # --- ASGI ------------------------------------------------------------------

    def router(self) -> "APIRouter":
        router = APIRouter()

        @router.post("/mcp")
        async def mcp_endpoint(request: Request):
            payload = await request.json()
            ctx = RunContext(session_id="mcp")
            if isinstance(payload, list):  # JSON-RPC batch
                return [await self.handle(p, ctx) for p in payload]
            return await self.handle(payload, ctx)

        return router


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
