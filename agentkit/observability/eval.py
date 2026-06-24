"""Evaluation runner (§7.2): score quality offline and push to the same hub.

Built-in metrics run offline with no extra services (keyword/heuristic judges);
DeepEval/Ragas/LLM-judge metrics layer on the same `Metric` surface. Scores attach
to the run's trace via the tracer (Langfuse when configured).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..types import RunResult
from .tracing import BaseTracer, NoOpTracer

if TYPE_CHECKING:
    from ..agent import Agent


class EvalItem(BaseModel):
    input: str
    expected: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class Score(BaseModel):
    name: str
    value: float
    comment: str | None = None


class EvalReport(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)  # mean per metric
    n: int = 0
    per_item: list[dict] = Field(default_factory=list)


@runtime_checkable
class Metric(Protocol):
    name: str

    async def score(self, item: EvalItem, output: RunResult) -> Score: ...


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class TaskSuccess:
    """Heuristic task success: expected answer present / strong token overlap."""

    name = "task_success"

    async def score(self, item: EvalItem, output: RunResult) -> Score:
        if not item.expected:
            return Score(name=self.name, value=1.0 if output.status == "done" else 0.0)
        exp, got = item.expected.lower(), (output.output or "").lower()
        if exp.strip() in got:
            return Score(name=self.name, value=1.0, comment="exact-substring")
        overlap = _tokens(exp) & _tokens(got)
        ratio = len(overlap) / max(1, len(_tokens(exp)))
        return Score(name=self.name, value=round(ratio, 3), comment="token-overlap")


class ToolCorrectness:
    """Did the run call the expected tools?"""

    name = "tool_correctness"

    async def score(self, item: EvalItem, output: RunResult) -> Score:
        if not item.expected_tools:
            return Score(name=self.name, value=1.0)
        used = {rec.name for rec in output.tool_calls}
        hit = sum(1 for t in item.expected_tools if t in used)
        return Score(name=self.name, value=round(hit / len(item.expected_tools), 3))


class Faithfulness:
    """Heuristic faithfulness: answer grounded in provided context tokens."""

    name = "faithfulness"

    async def score(self, item: EvalItem, output: RunResult) -> Score:
        if not item.context:
            return Score(name=self.name, value=1.0, comment="no-context")
        ctx_tokens = set().union(*[_tokens(c) for c in item.context])
        ans = _tokens(output.output) - {"the", "a", "an", "is", "of", "and", "to"}
        if not ans:
            return Score(name=self.name, value=0.0)
        grounded = len(ans & ctx_tokens) / len(ans)
        return Score(name=self.name, value=round(grounded, 3))


class Latency:
    name = "latency_ms"

    async def score(self, item: EvalItem, output: RunResult) -> Score:
        # Token-based proxy (latency captured upstream); reports total tokens.
        return Score(name=self.name, value=float(output.usage.total_tokens))


class EvalRunner:
    """Runs a dataset through a target (Agent/Flow) and scores each item."""

    def __init__(
        self,
        target: "Agent",
        *,
        dataset: list[EvalItem],
        metrics: list[Metric],
        tracer: BaseTracer | None = None,
    ) -> None:
        self.target = target
        self.dataset = dataset
        self.metrics = metrics
        self.tracer = tracer or getattr(target, "_tracer", None) or NoOpTracer()

    async def run(self, *, sample: int | None = None) -> EvalReport:
        items = self.dataset[:sample] if sample else self.dataset
        totals: dict[str, float] = {m.name: 0.0 for m in self.metrics}
        per_item: list[dict] = []
        for item in items:
            result: RunResult = await self.target.arun(item.input)
            row: dict[str, Any] = {"input": item.input, "output": result.output, "scores": {}}
            for metric in self.metrics:
                score = await metric.score(item, result)
                totals[metric.name] += score.value
                row["scores"][metric.name] = score.value
                if result.trace_url:  # push to the same trace when available
                    self.tracer.score(
                        result.session_id, score.name, score.value, score.comment
                    )
            per_item.append(row)
        self.tracer.flush()
        n = max(1, len(items))
        return EvalReport(
            metrics={k: round(v / n, 4) for k, v in totals.items()},
            n=len(items),
            per_item=per_item,
        )


DEFAULT_METRICS: list[Metric] = [TaskSuccess(), ToolCorrectness(), Latency()]
