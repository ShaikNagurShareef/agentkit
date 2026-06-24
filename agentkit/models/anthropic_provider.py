"""Anthropic provider via the official `anthropic` SDK.

Translates AgentKit's normalized Message/Tool types to/from Anthropic's native
message + tool-calling format (system param, tool_use / tool_result blocks).
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from ..context import RunContext
from ..errors import ModelError, RateLimitError
from ..tools.base import Tool
from ..types import Message, ToolCall, Usage
from .base import ModelInfo, ModelResponse, ModelSettings, StreamDelta

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the anthropic provider requires the 'anthropic' package. "
        "Install it with: pip install 'agentkit[anthropic]'"
    ) from e


DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider:
    """ModelProvider backed by `anthropic.AsyncAnthropic`."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = anthropic.AsyncAnthropic()

    # --- request translation ---------------------------------------------------

    def _to_anthropic_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert normalized messages to Anthropic's blocks format.

        System messages are pulled out separately by the caller. Assistant tool
        calls become `tool_use` blocks; tool results become `tool_result` blocks
        inside a user message.
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": str(m.content) if m.content is not None else "",
                            }
                        ],
                    }
                )
                continue
            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.args,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
                continue
            out.append({"role": m.role, "content": m.content or ""})
        return out

    def _tool_params(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters or {"type": "object", "properties": {}},
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
            "max_tokens": settings.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": self._to_anthropic_messages(messages),
        }
        if instructions:
            kwargs["system"] = instructions
        if tools:
            kwargs["tools"] = self._tool_params(tools)
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        if settings.top_p is not None:
            kwargs["top_p"] = settings.top_p
        if settings.stop:
            kwargs["stop_sequences"] = settings.stop
        return kwargs

    # --- response translation --------------------------------------------------

    def _parse_response(self, resp: Any) -> ModelResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, args=dict(block.input))
                )
        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
            total_tokens=getattr(resp.usage, "input_tokens", 0)
            + getattr(resp.usage, "output_tokens", 0),
        )
        message = Message(
            role="assistant",
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
        )
        return ModelResponse(message=message, tool_calls=tool_calls, usage=usage, raw=resp)

    # --- protocol methods ------------------------------------------------------

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
            resp = await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), where="model", cause="anthropic") from e
        except Exception as e:  # APIError, auth/config errors, etc.
            raise ModelError(str(e), where="model", cause=type(e).__name__) from e
        return self._parse_response(resp)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool],
        settings: ModelSettings,
        ctx: RunContext,
        instructions: str | None = None,
    ) -> AsyncIterator[StreamDelta]:
        kwargs = self._build_kwargs(messages, tools, settings, instructions)
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield StreamDelta(text=text)
                final = await stream.get_final_message()
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), where="model", cause="anthropic") from e
        except Exception as e:
            raise ModelError(str(e), where="model", cause=type(e).__name__) from e
        yield StreamDelta(final=self._parse_response(final))

    async def list_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        async for m in self._client.models.list():
            models.append(
                ModelInfo(
                    id=m.id,
                    provider="anthropic",
                    display_name=getattr(m, "display_name", None),
                )
            )
        return models


def build(model_id: str) -> AnthropicProvider:
    return AnthropicProvider(model_id)
