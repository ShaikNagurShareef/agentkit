"""M5 eval: EvalRunner + built-in metrics over a scripted target (no key)."""

from __future__ import annotations

import pytest

from agentkit import Agent, tool
from agentkit.models.base import ModelResponse, ModelSettings
from agentkit.observability import (
    ConsoleTracer,
    EvalItem,
    EvalRunner,
    NoOpTracer,
    TaskSuccess,
    ToolCorrectness,
    build_tracer,
)
from agentkit.types import Message, ToolCall, Usage


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ReactiveProvider:
    """Stateless: calls `add` once, then answers — works across many eval items."""

    model = "fake"

    async def complete(self, messages, *, tools, settings, ctx, instructions=None):
        has_result = any(m.role == "tool" for m in messages)
        if tools and not has_result:
            calls = [ToolCall(name="add", args={"a": 2, "b": 3})]
            return ModelResponse(
                message=Message(role="assistant", tool_calls=calls),
                tool_calls=calls,
                usage=Usage(total_tokens=5),
            )
        return ModelResponse(
            message=Message(role="assistant", content="the answer is 5"),
            tool_calls=[],
            usage=Usage(total_tokens=5),
        )

    async def stream(self, *a, **k):  # pragma: no cover
        yield None


@pytest.mark.asyncio
async def test_eval_runner_scores():
    agent = Agent(name="e", model="fake:fake", tools=[add])
    agent._provider = ReactiveProvider()
    dataset = [
        EvalItem(input="2+3?", expected="5", expected_tools=["add"]),
        EvalItem(input="add 2 and 3", expected="5", expected_tools=["add"]),
    ]
    runner = EvalRunner(agent, dataset=dataset, metrics=[TaskSuccess(), ToolCorrectness()])
    report = await runner.run()
    assert report.n == 2
    assert report.metrics["task_success"] == 1.0
    assert report.metrics["tool_correctness"] == 1.0
    assert len(report.per_item) == 2


def test_build_tracer_defaults():
    assert isinstance(build_tracer(None), NoOpTracer)
    assert isinstance(build_tracer("none"), NoOpTracer)
    assert isinstance(build_tracer("console"), ConsoleTracer)
    # "langfuse" with no keys configured falls back to no-op (zero infra)
    assert isinstance(build_tracer("langfuse"), NoOpTracer)
