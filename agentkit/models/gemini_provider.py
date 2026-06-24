"""Gemini provider via the official `google-genai` SDK."""

from __future__ import annotations

from typing import Any, AsyncIterator

from ..context import RunContext
from ..errors import ModelError
from ..tools.base import Tool
from ..types import Message, ToolCall, Usage
from .base import ModelInfo, ModelResponse, ModelSettings, StreamDelta

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the gemini provider requires the 'google-genai' package. "
        "Install it with: pip install 'agentkit[gemini]'"
    ) from e


# Gemini's FunctionDeclaration accepts only an OpenAPI-subset Schema; strip the
# JSON-Schema keys pydantic emits that it rejects ($schema, title, ...).
_SCHEMA_DROP = {"$schema", "title", "additionalProperties", "$defs", "definitions", "default"}


def _sanitize_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, val in schema.items():
        if key in _SCHEMA_DROP:
            continue
        if key == "properties" and isinstance(val, dict):
            out["properties"] = {k: _sanitize_schema(v) for k, v in val.items()}
        elif key == "items":
            out["items"] = _sanitize_schema(val)
        elif key in ("anyOf", "allOf", "oneOf") and isinstance(val, list):
            out[key] = [_sanitize_schema(v) for v in val]
        else:
            out[key] = val
    return out


class GeminiProvider:
    """ModelProvider backed by `google-genai`.

    Gemini matches tool results to calls by function name (no call ids), so we
    key tool messages by their `name`.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = genai.Client()

    def _to_contents(self, messages: list[Message]) -> list[Any]:
        contents: list[Any] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part.from_function_response(
                                name=m.name or "tool",
                                response={"result": m.content},
                            )
                        ],
                    )
                )
            elif m.role == "assistant" and m.tool_calls:
                parts = []
                if m.content:
                    parts.append(genai_types.Part(text=m.content))
                for tc in m.tool_calls:
                    parts.append(
                        genai_types.Part.from_function_call(name=tc.name, args=tc.args)
                    )
                contents.append(genai_types.Content(role="model", parts=parts))
            else:
                role = "model" if m.role == "assistant" else "user"
                contents.append(
                    genai_types.Content(role=role, parts=[genai_types.Part(text=m.content or "")])
                )
        return contents

    def _build_config(
        self, tools: list[Tool], settings: ModelSettings, instructions: str | None
    ) -> Any:
        cfg: dict[str, Any] = {}
        if instructions:
            cfg["system_instruction"] = instructions
        if settings.temperature is not None:
            cfg["temperature"] = settings.temperature
        if settings.top_p is not None:
            cfg["top_p"] = settings.top_p
        if settings.max_tokens is not None:
            cfg["max_output_tokens"] = settings.max_tokens
        if settings.stop:
            cfg["stop_sequences"] = settings.stop
        if tools:
            cfg["tools"] = [
                genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters=_sanitize_schema(
                                t.parameters or {"type": "object", "properties": {}}
                            ),
                        )
                        for t in tools
                    ]
                )
            ]
        return genai_types.GenerateContentConfig(**cfg)

    def _parse(self, resp: Any) -> ModelResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:  # e.g. MALFORMED_FUNCTION_CALL / safety: empty candidate
                continue
            for part in (content.parts or []):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc:
                    tool_calls.append(
                        ToolCall(name=fc.name, args=dict(fc.args) if fc.args else {})
                    )
        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
            total_tokens=getattr(um, "total_token_count", 0) or 0,
        )
        message = Message(
            role="assistant", content="".join(text_parts) or None, tool_calls=tool_calls
        )
        return ModelResponse(message=message, tool_calls=tool_calls, usage=usage, raw=resp)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[Tool],
        settings: ModelSettings,
        ctx: RunContext,
        instructions: str | None = None,
    ) -> ModelResponse:
        contents = self._to_contents(messages)
        config = self._build_config(tools, settings, instructions)
        result: ModelResponse | None = None
        # gemini-2.5-flash occasionally returns an empty/malformed candidate
        # (MALFORMED_FUNCTION_CALL); retry once before giving up.
        for _ in range(2):
            try:
                resp = await self._client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as e:  # google-genai raises various errors
                raise ModelError(str(e), where="model", cause="gemini") from e
            result = self._parse(resp)
            if result.message.content or result.tool_calls:
                return result
        return result  # type: ignore[return-value]

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool],
        settings: ModelSettings,
        ctx: RunContext,
        instructions: str | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """True token streaming via google-genai's streaming endpoint."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = Usage()
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=self._to_contents(messages),
                config=self._build_config(tools, settings, instructions),
            )
            async for chunk in stream:
                if getattr(chunk, "text", None):
                    text_parts.append(chunk.text)
                    yield StreamDelta(text=chunk.text)
                for cand in getattr(chunk, "candidates", None) or []:
                    for part in (cand.content.parts or []) if cand.content else []:
                        fc = getattr(part, "function_call", None)
                        if fc:
                            tool_calls.append(
                                ToolCall(name=fc.name, args=dict(fc.args) if fc.args else {})
                            )
                um = getattr(chunk, "usage_metadata", None)
                if um:
                    usage = Usage(
                        input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                        total_tokens=getattr(um, "total_token_count", 0) or 0,
                    )
        except Exception as e:  # noqa: BLE001
            raise ModelError(str(e), where="model", cause="gemini") from e
        message = Message(
            role="assistant", content="".join(text_parts) or None, tool_calls=tool_calls
        )
        yield StreamDelta(final=ModelResponse(message=message, tool_calls=tool_calls, usage=usage))

    async def list_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        async for m in await self._client.aio.models.list():
            models.append(
                ModelInfo(
                    id=getattr(m, "name", ""),
                    provider="gemini",
                    display_name=getattr(m, "display_name", None),
                )
            )
        return models


def build(model_id: str) -> GeminiProvider:
    return GeminiProvider(model_id)
