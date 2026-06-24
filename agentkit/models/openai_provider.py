"""OpenAI provider via the official `openai` SDK (Chat Completions + tools)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..context import RunContext
from ..errors import ModelError, RateLimitError
from ..tools.base import Tool
from ..types import Message, ToolCall, Usage
from .base import ModelInfo, ModelResponse, ModelSettings, StreamDelta

try:
    import openai
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the openai provider requires the 'openai' package. "
        "Install it with: pip install 'agentkit[openai]'"
    ) from e


class OpenAIProvider:
    """ModelProvider backed by `openai.AsyncOpenAI` (also usable for compatible APIs)."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = openai.AsyncOpenAI()

    def _to_openai_messages(
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

    def _build_kwargs(
        self,
        messages: list[Message],
        tools: list[Tool],
        settings: ModelSettings,
        instructions: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(messages, instructions),
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
        return kwargs

    def _parse_message(self, msg: Any, usage_obj: Any) -> ModelResponse:
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
        kwargs = self._build_kwargs(messages, tools, settings, instructions)
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), where="model", cause="openai") from e
        except openai.OpenAIError as e:
            raise ModelError(str(e), where="model", cause="openai") from e
        return self._parse_message(resp.choices[0].message, resp.usage)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool],
        settings: ModelSettings,
        ctx: RunContext,
        instructions: str | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """True token streaming with incremental tool-call assembly."""
        kwargs = self._build_kwargs(messages, tools, settings, instructions)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        content_parts: list[str] = []
        acc: dict[int, dict] = {}
        usage_obj = None
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage_obj = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    yield StreamDelta(text=delta.content)
                for tc in delta.tool_calls or []:
                    slot = acc.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), where="model", cause="openai") from e
        except openai.OpenAIError as e:
            raise ModelError(str(e), where="model", cause="openai") from e

        tool_calls = []
        for idx in sorted(acc):
            slot = acc[idx]
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=slot["id"] or "", name=slot["name"], args=args))
        message = Message(
            role="assistant", content="".join(content_parts) or None, tool_calls=tool_calls
        )
        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
        )
        yield StreamDelta(final=ModelResponse(message=message, tool_calls=tool_calls, usage=usage))

    async def list_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        resp = await self._client.models.list()
        for m in resp.data:
            models.append(ModelInfo(id=m.id, provider="openai", created=getattr(m, "created", None)))
        return models


def build(model_id: str) -> OpenAIProvider:
    return OpenAIProvider(model_id)
