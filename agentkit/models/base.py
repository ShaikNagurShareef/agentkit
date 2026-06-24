"""Model provider abstraction (official-SDK-backed).

The engine speaks only to the ``ModelProvider`` protocol. Concrete providers wrap
each vendor's official SDK and adapt our normalized Message/Tool types to and from
the provider's native tool-calling format.

Model strings are ``"<provider>:<model_id>"`` (e.g. ``"anthropic:claude-opus-4-8"``).
A bare id resolves against ``AGENTKIT_DEFAULT_PROVIDER`` (default: anthropic).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ..context import RunContext
from ..tools.base import Tool
from ..types import Message, ToolCall, Usage


class ModelSettings(BaseModel):
    """Generation settings, mapped per provider where supported."""

    model_config = ConfigDict(extra="allow")

    temperature: float | None = None
    max_tokens: int | None = 4096
    top_p: float | None = None
    stop: list[str] | None = None


class ModelInfo(BaseModel):
    """A model discovered live from a provider's API."""

    id: str
    provider: str
    display_name: str | None = None
    created: int | None = None


@dataclass
class ModelResponse:
    """Normalized result of one model call."""

    message: Message  # assistant message (content and/or tool_calls)
    tool_calls: list[ToolCall]
    usage: Usage
    raw: Any = None


@dataclass
class StreamDelta:
    """A streaming increment: a token of text and/or a completed assistant message."""

    text: str | None = None
    final: ModelResponse | None = None


@runtime_checkable
class ModelProvider(Protocol):
    """Provider contract the engine depends on."""

    model: str  # bare model id (no provider prefix)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[Tool],
        settings: ModelSettings,
        ctx: RunContext,
        instructions: str | None = None,
    ) -> ModelResponse: ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[Tool],
        settings: ModelSettings,
        ctx: RunContext,
        instructions: str | None = None,
    ) -> AsyncIterator[StreamDelta]: ...

    async def list_models(self) -> list[ModelInfo]:
        """Query the provider API for currently available models."""
        ...


# provider name -> import path of a factory ``def build(model_id) -> ModelProvider``
_PROVIDERS: dict[str, str] = {
    "anthropic": "agentkit.models.anthropic_provider:build",
    "openai": "agentkit.models.openai_provider:build",
    "gemini": "agentkit.models.gemini_provider:build",
    "google": "agentkit.models.gemini_provider:build",
    "groq": "agentkit.models.groq_provider:build",
}


def split_model(model: str) -> tuple[str, str]:
    """Split ``"provider:model_id"``; bare id uses the default provider."""
    if ":" in model:
        provider, _, model_id = model.partition(":")
        return provider.strip().lower(), model_id.strip()
    default = os.environ.get("AGENTKIT_DEFAULT_PROVIDER", "anthropic").lower()
    return default, model.strip()


def resolve_provider(model: str, settings: ModelSettings | None = None) -> ModelProvider:
    """Resolve a model string to a concrete provider instance (lazy SDK import)."""
    provider, model_id = split_model(model)
    if provider not in _PROVIDERS:
        raise ValueError(
            f"unknown model provider '{provider}'. "
            f"Known: {', '.join(sorted(set(_PROVIDERS)))}"
        )
    module_path, _, attr = _PROVIDERS[provider].partition(":")
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:  # missing optional extra
        raise ImportError(
            f"provider '{provider}' requires an optional dependency. "
            f"Install it with: pip install 'agentkit[{provider}]'  ({e})"
        ) from e
    build = getattr(module, attr)
    return build(model_id)
