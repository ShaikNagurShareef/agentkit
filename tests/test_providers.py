"""Offline provider translation tests (no network).

These validate that each provider correctly maps AgentKit's normalized
Message/Tool types to/from the vendor's native format. They run only when the
provider's SDK extra is installed and use a dummy key (no requests are made).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from agentkit import tool
from agentkit.context import RunContext
from agentkit.models.base import ModelSettings
from agentkit.types import Message, ToolCall


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_anthropic_translation_and_parse():
    anthropic = pytest.importorskip("anthropic")
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-for-offline-test")
    from agentkit.models.anthropic_provider import AnthropicProvider

    p = AnthropicProvider("claude-opus-4-8")

    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="add", args={"a": 1, "b": 2})]),
        Message(role="tool", tool_call_id="c1", name="add", content="3"),
    ]
    anth = p._to_anthropic_messages(messages)
    assert anth[0] == {"role": "user", "content": "hi"}
    assert anth[1]["role"] == "assistant"
    assert anth[1]["content"][0]["type"] == "tool_use"
    assert anth[2]["content"][0]["type"] == "tool_result"
    assert anth[2]["content"][0]["tool_use_id"] == "c1"

    tparams = p._tool_params([add])
    assert tparams[0]["name"] == "add"
    assert "input_schema" in tparams[0]

    # Parse a fake Anthropic response object (tool_use + usage).
    fake = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="let me add"),
            SimpleNamespace(type="tool_use", id="c2", name="add", input={"a": 4, "b": 5}),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    resp = p._parse_response(fake)
    assert resp.message.content == "let me add"
    assert resp.tool_calls[0].name == "add"
    assert resp.tool_calls[0].args == {"a": 4, "b": 5}
    assert resp.usage.total_tokens == 18


def test_openai_translation_and_parse():
    pytest.importorskip("openai")
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-for-offline-test")
    from agentkit.models.openai_provider import OpenAIProvider

    p = OpenAIProvider("gpt-4o")
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="add", args={"a": 1, "b": 2})]),
        Message(role="tool", tool_call_id="c1", name="add", content="3"),
    ]
    oai = p._to_openai_messages(messages, instructions="be helpful")
    assert oai[0] == {"role": "system", "content": "be helpful"}
    assert oai[2]["tool_calls"][0]["function"]["name"] == "add"
    assert oai[3]["role"] == "tool" and oai[3]["tool_call_id"] == "c1"

    fake_msg = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="c9",
                function=SimpleNamespace(name="add", arguments='{"a": 6, "b": 7}'),
            )
        ],
    )
    fake_usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    resp = p._parse_message(fake_msg, fake_usage)
    assert resp.tool_calls[0].name == "add"
    assert resp.tool_calls[0].args == {"a": 6, "b": 7}
    assert resp.usage.total_tokens == 7


def test_resolve_provider_unknown():
    from agentkit.models.base import resolve_provider

    with pytest.raises(ValueError):
        resolve_provider("nope:some-model")


def test_split_model_default_and_explicit():
    from agentkit.models.base import split_model

    assert split_model("anthropic:claude-opus-4-8") == ("anthropic", "claude-opus-4-8")
    provider, model_id = split_model("gpt-4o")
    assert model_id == "gpt-4o"  # provider falls back to default


def test_gemini_schema_sanitizer_strips_unsupported_keys():
    # Gemini's FunctionDeclaration rejects $schema/title/additionalProperties;
    # the sanitizer must drop them recursively while keeping the shape.
    pytest.importorskip("google.genai")
    from agentkit.models.gemini_provider import _sanitize_schema

    raw = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "add_Args",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": "integer", "title": "A", "default": 0},
            "items": {"type": "array", "items": {"type": "string", "title": "X"}},
        },
        "required": ["a"],
    }
    clean = _sanitize_schema(raw)
    assert "$schema" not in clean and "title" not in clean and "additionalProperties" not in clean
    assert clean["type"] == "object"
    assert clean["required"] == ["a"]
    assert clean["properties"]["a"] == {"type": "integer"}  # title/default stripped
    assert clean["properties"]["items"]["items"] == {"type": "string"}  # recursed
