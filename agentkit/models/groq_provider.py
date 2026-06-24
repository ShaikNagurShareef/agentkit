"""Groq provider via the official `groq` SDK (OpenAI-style chat + tools)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..context import RunContext
from ..errors import ModelError, RateLimitError
from ..tools.base import Tool
from ..types import Message, ToolCall, Usage
from .base import ModelInfo, ModelResponse, ModelSettings, StreamDelta

try:
    import groq
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the groq provider requires the 'groq' package. "
        "Install it with: pip install 'agentkit[groq]'"
    ) from e


class GroqProvider:
    """ModelProvider backed by `groq.AsyncGroq`."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = groq.AsyncGroq()

    def _to_messages(
        self, messages: list[Message], instructions: str | None
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if instructions:
            out.append({"role": "system", "content": instructions})
        for m in messages:
            if m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": str(m.content) if m.content is not None else "",
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.args),
                                },
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    def _tool_params(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

    def _parse(self, msg: Any, usage_obj: Any) -> ModelResponse:
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
        )
        message = Message(role="assistant", content=msg.content, tool_calls=tool_calls)
        return ModelResponse(message=message, tool_calls=tool_calls, usage=usage, raw=msg)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[Tool],
        settings: ModelSettings,
        ctx: RunContext,
        instructions: str | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_messages(messages, instructions),
        }
        if settings.max_tokens is not None:
            kwargs["max_tokens"] = settings.max_tokens
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        if settings.top_p is not None:
            kwargs["top_p"] = settings.top_p
        if settings.stop:
            kwargs["stop"] = settings.stop
        if tools:
            kwargs["tools"] = self._tool_params(tools)
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except groq.RateLimitError as e:
            raise RateLimitError(str(e), where="model", cause="groq") from e
        except groq.GroqError as e:
            raise ModelError(str(e), where="model", cause="groq") from e
        return self._parse(resp.choices[0].message, resp.usage)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool],
        settings: ModelSettings,
        ctx: RunContext,
        instructions: str | None = None,
    ) -> AsyncIterator[StreamDelta]:
        resp = await self.complete(
            messages, tools=tools, settings=settings, ctx=ctx, instructions=instructions
        )
        if resp.message.content:
            yield StreamDelta(text=resp.message.content)
        yield StreamDelta(final=resp)

    async def list_models(self) -> list[ModelInfo]:
        resp = await self._client.models.list()
        return [ModelInfo(id=m.id, provider="groq", created=getattr(m, "created", None)) for m in resp.data]


def build(model_id: str) -> GroqProvider:
    return GroqProvider(model_id)
