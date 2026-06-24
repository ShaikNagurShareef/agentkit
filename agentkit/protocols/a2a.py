"""A2A (§4.3): publish the agent as an A2A peer and call other A2A peers.

The agent exposes an Agent Card at ``/.well-known/agent.json`` and a task endpoint
at ``/a2a`` (``tasks/send`` / ``tasks/get``). Remote peers listed on an Agent
become callable tools (``source="a2a"``): the model "calls" the peer as a tool and
the call is a ``tasks/send`` under the hood.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..context import RunContext
from ..errors import A2ATaskFailed
from ..identity.secrets import AuthConfig
from ..tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from ..agent import Agent


class Skill(BaseModel):
    id: str
    name: str
    description: str = ""
    examples: list[str] = Field(default_factory=list)


class Capabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False


class AgentCard(BaseModel):
    name: str
    description: str = ""
    url: str = ""
    version: str = "0.1.0"
    capabilities: Capabilities = Field(default_factory=Capabilities)
    skills: list[Skill] = Field(default_factory=list)
    auth: dict = Field(default_factory=dict)


def _text_of(message: dict) -> str:
    parts = message.get("parts", []) if isinstance(message, dict) else []
    return "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _text_message(text: str, role: str = "agent") -> dict:
    return {"role": role, "parts": [{"type": "text", "text": text}]}


class A2AServer:
    """Serves an Agent as an A2A peer."""

    def __init__(self, agent: "Agent", *, card: AgentCard | None = None) -> None:
        self.agent = agent
        self.card = card or AgentCard(
            name=agent.name,
            description=agent.instructions or "",
            skills=[Skill(id="default", name=agent.name, description=agent.instructions or "")],
        )
        self._tasks: dict[str, dict] = {}

    def router(self) -> "APIRouter":
        router = APIRouter()

        @router.get("/.well-known/agent.json")
        async def agent_card():
            return self.card.model_dump()

        @router.post("/a2a")
        async def a2a(request: Request):
            payload = await request.json()
            method = payload.get("method")
            params = payload.get("params") or {}
            rid = payload.get("id")
            if method == "tasks/send":
                task = await self._handle_send(params)
                return {"jsonrpc": "2.0", "id": rid, "result": task}
            if method == "tasks/get":
                task = self._tasks.get(params.get("id"))
                if task is None:
                    raise HTTPException(status_code=404, detail="task not found")
                return {"jsonrpc": "2.0", "id": rid, "result": task}
            raise HTTPException(status_code=400, detail=f"unknown method '{method}'")

        return router

    async def _handle_send(self, params: dict) -> dict:
        task_id = params.get("id") or f"task_{uuid.uuid4().hex[:12]}"
        text = _text_of(params.get("message", {}))
        self._tasks[task_id] = {"id": task_id, "status": {"state": "working"}, "artifacts": []}
        result = await self.agent._arun_result(text, session_id=task_id, deadline_s=None)
        state = "completed" if result.status == "done" else "failed"
        task = {
            "id": task_id,
            "status": {"state": state},
            "artifacts": [{"parts": [{"type": "text", "text": result.output}]}],
        }
        self._tasks[task_id] = task
        return task


class A2AClient:
    """Call remote A2A peers."""

    def __init__(self, *, auth: AuthConfig | None = None) -> None:
        self.auth = auth
        self._ids = 0

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.auth:
            h.update(self.auth.headers())
        return h

    async def get_card(self, peer_url: str) -> AgentCard:
        base = peer_url.rstrip("/")
        url = base if base.endswith("agent.json") else f"{base}/.well-known/agent.json"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            return AgentCard(**resp.json())

    async def send(self, peer_url: str, message: str, *, stream: bool = False) -> dict:
        a2a_url = self._task_url(peer_url)
        self._ids += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._ids,
            "method": "tasks/send",
            "params": {"message": _text_message(message, role="user")},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(a2a_url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        if "error" in data:
            raise A2ATaskFailed(data["error"].get("message", "a2a error"), where=peer_url)
        return data["result"]

    async def get(self, peer_url: str, task_id: str) -> dict:
        a2a_url = self._task_url(peer_url)
        self._ids += 1
        payload = {"jsonrpc": "2.0", "id": self._ids, "method": "tasks/get", "params": {"id": task_id}}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(a2a_url, json=payload, headers=self._headers())
            resp.raise_for_status()
            return resp.json()["result"]

    @staticmethod
    def _task_url(peer_url: str) -> str:
        base = peer_url.rstrip("/")
        if base.endswith("/.well-known/agent.json"):
            base = base[: -len("/.well-known/agent.json")]
        if base.endswith("/a2a"):
            return base
        return f"{base}/a2a"


def artifact_text(task: dict) -> str:
    """Extract the text artifact from a completed A2A task."""
    for art in task.get("artifacts", []):
        for part in art.get("parts", []):
            if part.get("type") == "text":
                return part.get("text", "")
    return ""


async def peer_as_tool(peer_url: str, *, auth: AuthConfig | None = None) -> Tool:
    """Build a callable Tool that delegates to a remote A2A peer (FR-7)."""
    client = A2AClient(auth=auth)
    try:
        card = await client.get_card(peer_url)
        name = card.name
        description = card.description or f"Delegate a task to the '{name}' agent."
    except Exception:
        name = peer_url
        description = f"Delegate a task to the remote agent at {peer_url}."

    async def handler(args: dict, ctx: RunContext) -> ToolResult:
        task = await client.send(peer_url, args.get("input", ""))
        if task.get("status", {}).get("state") != "completed":
            return ToolResult(ok=False, content=None, error=A2ATaskFailed("peer task failed", where=peer_url).info)
        return ToolResult(ok=True, content=artifact_text(task))

    safe = "".join(c if c.isalnum() else "_" for c in name)[:40] or "peer"
    return Tool(
        name=f"delegate_{safe}",
        description=description,
        parameters={
            "type": "object",
            "properties": {"input": {"type": "string", "description": "the task for the peer agent"}},
            "required": ["input"],
        },
        source="a2a",
        timeout_s=120,
        handler=handler,
    )
