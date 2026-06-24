"""Browser automation (§6.1).

Wraps `browser-use` over a local Chromium (Playwright) — no VM. ``backend="steel"``
swaps in a self-hosted/remote browser API for parallel sessions. DOM is distilled
to indexed interactive elements before being handed to the model.

The dependency is optional (`agentkit[browser]`); ``run_task`` fails with a clear
error if it is not installed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ..context import RunContext
from ..errors import ToolError
from .base import Tool, ToolResult


class BrowserResult(BaseModel):
    output: str = ""
    steps: int = 0
    final_url: str | None = None


class BrowserTool:
    def __init__(
        self,
        *,
        headless: bool = True,
        profile: str | None = None,
        backend: Literal["local", "steel"] = "local",
    ) -> None:
        self.headless = headless
        self.profile = profile
        self.backend = backend

    async def run_task(self, instruction: str, *, ctx: RunContext) -> BrowserResult:
        try:
            from browser_use import Agent as BrowserAgent  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dep
            raise ToolError(
                "browser automation needs browser-use: pip install 'agentkit[browser]'",
                where="browser",
            ) from e
        # browser-use drives its own model; we pass the instruction through.
        agent = BrowserAgent(task=instruction)  # pragma: no cover - needs browser
        history = await agent.run()
        return BrowserResult(output=str(history.final_result() or ""), steps=len(history.history))

    def as_tool(self) -> Tool:
        async def handler(args: dict, ctx: RunContext) -> ToolResult:
            result = await self.run_task(args.get("instruction", ""), ctx=ctx)
            return ToolResult(ok=True, content=result.output, raw=result.model_dump())

        return Tool(
            name="browser",
            description="Drive a web browser to navigate, click, type, extract, and screenshot.",
            parameters={
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "what to do in the browser"}
                },
                "required": ["instruction"],
            },
            source="local",
            timeout_s=300,
            handler=handler,
        )
