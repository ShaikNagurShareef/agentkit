"""Shared test fixtures: a scriptable FakeModelProvider needing no API key."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from agentkit.context import RunContext
from agentkit.models.base import ModelInfo, ModelResponse, ModelSettings, StreamDelta
from agentkit.tools.base import Tool
from agentkit.types import Message, ToolCall, Usage


class FakeModelProvider:
    """Returns scripted responses so the full graph loop runs deterministically.

    ``script`` is a list of either:
      * a list[ToolCall] -> assistant message requesting those tools, or
      * a str           -> final assistant message with that text.
    """

    model = "fake"

    def __init__(self, script: list):
        self.script = list(script)
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[Tool],
        settings: ModelSettings,
        ctx: RunContext,
        instructions: str | None = None,
    ) -> ModelResponse:
        step = self.script[self.calls] if self.calls < len(self.script) else "done"
        self.calls += 1
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        if isinstance(step, list):
            tool_calls = list(step)
            msg = Message(role="assistant", content=None, tool_calls=tool_calls)
            return ModelResponse(message=msg, tool_calls=tool_calls, usage=usage)
        msg = Message(role="assistant", content=str(step))
        return ModelResponse(message=msg, tool_calls=[], usage=usage)

    async def stream(self, messages, *, tools, settings, ctx, instructions=None) -> AsyncIterator[StreamDelta]:
        resp = await self.complete(
            messages, tools=tools, settings=settings, ctx=ctx, instructions=instructions
        )
        if resp.message.content:
            yield StreamDelta(text=resp.message.content)
        yield StreamDelta(final=resp)

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="fake", provider="fake")]


@pytest.fixture
def make_agent():
    """Factory: build an Agent wired to a FakeModelProvider with a given script."""
    from agentkit import Agent

    def _factory(script, **kwargs):
        agent = Agent(name="t", model="fake:fake", **kwargs)
        agent._provider = FakeModelProvider(script)
        return agent

    return _factory


def tool_call(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args)
