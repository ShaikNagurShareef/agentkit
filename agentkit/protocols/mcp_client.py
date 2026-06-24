"""MCP client (§4.1): consume an external MCP server and surface its tools.

Connects over Streamable HTTP (JSON-RPC), discovers tools on connect, and adapts
them to AgentKit ``Tool`` objects namespaced as ``{namespace}.{tool}`` to avoid
collisions. Each ``call_tool`` opens a tool span (observability wires in M5).
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from ..context import RunContext
from ..errors import MCPConnectionError, ToolError
from ..identity.secrets import AuthConfig
from ..tools.base import Tool, ToolResult


class MCPToolDescriptor(BaseModel):
    name: str
    description: str = ""
    inputSchema: dict = {}


class MCPClient:
    def __init__(
        self,
        url: str,
        *,
        transport: Literal["http", "stdio", "sse"] = "http",
        auth: AuthConfig | None = None,
        namespace: str | None = None,
    ) -> None:
        if transport != "http":
            raise NotImplementedError(f"MCP transport '{transport}' not supported yet (use http)")
        self.url = url
        self.auth = auth
        self.namespace = namespace
        self._ids = itertools.count(1)
        self._descriptors: list[MCPToolDescriptor] = []
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.auth:
            headers.update(self.auth.headers())
        return headers

    async def _rpc(self, method: str, params: dict | None = None) -> Any:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params or {}}
        try:
            resp = await self._client.post(self.url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise MCPConnectionError(str(e), where=self.url) from e
        if "error" in data:
            raise ToolError(data["error"].get("message", "mcp error"), where=self.url)
        return data.get("result")

    async def connect(self) -> None:
        await self._rpc("initialize", {"protocolVersion": "2025-06-18"})
        result = await self._rpc("tools/list")
        self._descriptors = [MCPToolDescriptor(**d) for d in result.get("tools", [])]

    async def list_tools(self) -> list[MCPToolDescriptor]:
        if not self._descriptors:
            await self.connect()
        return self._descriptors

    async def call_tool(self, name: str, args: dict, *, ctx: RunContext) -> ToolResult:
        result = await self._rpc("tools/call", {"name": name, "arguments": args})
        is_error = bool(result.get("isError"))
        content_blocks = result.get("content", [])
        text = "\n".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        # Try to decode JSON payloads back into structured content.
        content: Any = text
        try:
            content = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        if is_error:
            return ToolResult(ok=False, content=None, error=ToolError(text, where=name).info)
        return ToolResult(ok=True, content=content)

    def as_tools(self) -> list[Tool]:
        """Adapt discovered MCP tools into namespaced AgentKit Tool objects."""
        tools: list[Tool] = []
        for d in self._descriptors:
            qualified = f"{self.namespace}.{d.name}" if self.namespace else d.name
            tools.append(self._make_tool(qualified, d))
        return tools

    def _make_tool(self, qualified: str, d: MCPToolDescriptor) -> Tool:
        remote_name = d.name

        async def handler(args: dict, ctx: RunContext) -> ToolResult:
            return await self.call_tool(remote_name, args, ctx=ctx)

        return Tool(
            name=qualified,
            description=d.description or remote_name,
            parameters=d.inputSchema or {"type": "object", "properties": {}},
            source="mcp",
            handler=handler,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
