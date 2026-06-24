"""Flow engine (§3.2): compose agents/functions with branching, parallelism, loops.

A Flow is a graph of nodes (agents, functions, or nested flows) with sequential,
conditional, parallel, loop, and map control. It exposes the same run/serve surface
as an Agent, so a flow is served as one app (UC-3).

Implementation note: M7 ships a native async flow executor behind the
``CompiledGraph`` seam (§14). The declarative ``FlowSpec`` matches the design so a
LangGraph-backed compiler can be swapped in without changing flow-authoring code.
"""

from __future__ import annotations

import inspect
import uuid
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from .types import (
    DoneEvent,
    ErrorEvent,
    Message,
    NodeEndEvent,
    NodeStartEvent,
    RunResult,
    Usage,
)

Node = Any  # Agent | Callable | Flow


def _node_kind(node: Node) -> str:
    from .agent import Agent

    if isinstance(node, Agent):
        return "agent"
    if isinstance(node, Flow):
        return "flow"
    return "function"


# --- declarative spec (§3.2.2) -------------------------------------------------


class NodeSpec(BaseModel):
    id: str
    kind: Literal["agent", "function", "flow"]
    ref: str  # import path ("module:attr") or relative spec file ("./x.yaml")
    config: dict = Field(default_factory=dict)


class EdgeSpec(BaseModel):
    src: str
    dst: str | list[str]  # list = parallel fan-out
    condition: str | None = None  # python expr over {outputs, input}


class FlowSpec(BaseModel):
    name: str
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    entry: str
    finish: str | list[str]


# --- execution steps -----------------------------------------------------------


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, RunResult):
        return value.output
    return str(value)


async def _run_node(node: Node, value: Any, session_id: str) -> tuple[Any, list[Message], Usage]:
    from .agent import Agent

    if isinstance(node, Agent):
        r = await node._arun_result(_as_text(value), session_id=session_id, deadline_s=None)
        return r.output, r.messages, r.usage
    if isinstance(node, Flow):
        r = await node.arun(value, session_id=session_id)
        return r.output, r.messages, r.usage
    out = node(value)
    if inspect.isawaitable(out):
        out = await out
    return out, [], Usage()


class _Step:
    async def execute(self, ctx: "_FlowRun", value: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _Single(_Step):
    def __init__(self, node: Node, name: str) -> None:
        self.node = node
        self.name = name

    async def execute(self, ctx: "_FlowRun", value: Any) -> Any:
        out, messages, usage = await _run_node(self.node, value, ctx.session_id)
        ctx.record(self.name, out, messages, usage)
        return out


class _Parallel(_Step):
    def __init__(self, branches: list[tuple[str, Node]]) -> None:
        self.branches = branches

    async def execute(self, ctx: "_FlowRun", value: Any) -> Any:
        import asyncio

        results = await asyncio.gather(
            *[_run_node(node, value, ctx.session_id) for _, node in self.branches]
        )
        merged: dict[str, Any] = {}
        for (name, _), (out, messages, usage) in zip(self.branches, results):
            ctx.record(name, out, messages, usage)
            merged[name] = out
        return merged


class _Conditional(_Step):
    def __init__(self, predicate: Callable[[Any], bool], then: _Step, otherwise: _Step | None) -> None:
        self.predicate = predicate
        self.then = then
        self.otherwise = otherwise

    async def execute(self, ctx: "_FlowRun", value: Any) -> Any:
        take = self.predicate(value)
        branch = self.then if take else self.otherwise
        if branch is None:
            return value
        return await branch.execute(ctx, value)


class _Loop(_Step):
    def __init__(self, body: _Step, until: Callable[[Any], bool], max_iter: int) -> None:
        self.body = body
        self.until = until
        self.max_iter = max_iter

    async def execute(self, ctx: "_FlowRun", value: Any) -> Any:
        for _ in range(self.max_iter):
            value = await self.body.execute(ctx, value)
            if self.until(value):
                break
        return value


class _Map(_Step):
    def __init__(self, body: _Step, name: str) -> None:
        self.body = body
        self.name = name

    async def execute(self, ctx: "_FlowRun", value: Any) -> Any:
        import asyncio

        items = value if isinstance(value, list) else [value]
        results = await asyncio.gather(*[self.body.execute(ctx, item) for item in items])
        return list(results)


class _FlowRun:
    """Mutable accumulator for one flow invocation."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.node_outputs: dict[str, Any] = {}
        self.messages: list[Message] = []
        self.usage = Usage()

    def record(self, name: str, output: Any, messages: list[Message], usage: Usage) -> None:
        self.node_outputs[name] = output
        self.messages.extend(messages)
        self.usage = self.usage + usage


# --- builder API (§3.2.1) ------------------------------------------------------


class ConditionalBuilder:
    def __init__(self, flow: "Flow", predicate: Callable[[Any], bool]) -> None:
        self._flow = flow
        self._predicate = predicate
        self._then: _Step | None = None

    def then(self, node: Node, *, name: str | None = None) -> "ConditionalBuilder":
        self._then = _Single(node, name or self._flow._auto_name(node))
        return self

    def otherwise(self, node: Node, *, name: str | None = None) -> "Flow":
        other = _Single(node, name or self._flow._auto_name(node))
        assert self._then is not None, "call .then(...) before .otherwise(...)"
        self._flow._steps.append(_Conditional(self._predicate, self._then, other))
        return self._flow

    def end(self) -> "Flow":
        """Finish a conditional with no else-branch."""
        assert self._then is not None
        self._flow._steps.append(_Conditional(self._predicate, self._then, None))
        return self._flow


class Flow:
    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: list[_Step] = []
        self._counter = 0
        self._memory_cfg = None
        self._tracer = None
        self._memory_store = None

    def _auto_name(self, node: Node) -> str:
        self._counter += 1
        base = getattr(node, "name", None) or getattr(node, "__name__", "node")
        return f"{base}_{self._counter}"

    def step(self, node: Node, *, name: str | None = None) -> "Flow":
        self._steps.append(_Single(node, name or self._auto_name(node)))
        return self

    def parallel(self, *nodes: Node) -> "Flow":
        branches = [(self._auto_name(n), n) for n in nodes]
        self._steps.append(_Parallel(branches))
        return self

    def when(self, predicate: Callable[[Any], bool]) -> ConditionalBuilder:
        return ConditionalBuilder(self, predicate)

    def loop(self, body: Node, *, until: Callable[[Any], bool], max_iter: int = 10) -> "Flow":
        self._steps.append(_Loop(_Single(body, self._auto_name(body)), until, max_iter))
        return self

    def map(self, body: Node, *, over: str | None = None) -> "Flow":
        self._steps.append(_Map(_Single(body, self._auto_name(body)), self._auto_name(body)))
        return self

    # --- execution -------------------------------------------------------------

    async def arun(self, input: Any, *, session_id: str | None = None, **kw: Any) -> RunResult:
        sid = session_id or f"flow_{uuid.uuid4().hex[:12]}"
        run = _FlowRun(sid)
        value: Any = input
        try:
            for step in self._steps:
                value = await step.execute(run, value)
        except Exception as e:  # noqa: BLE001 — surface as a typed error result
            from .errors import ErrorInfo

            return RunResult(
                output="",
                session_id=sid,
                status="error",
                error=ErrorInfo(type="AgentError", message=str(e), where="flow",
                                cause=type(e).__name__),
            )
        return RunResult(
            output=_as_text(value),
            messages=run.messages,
            session_id=sid,
            status="done",
            usage=run.usage,
        )

    async def astream(self, input: Any, *, session_id: str | None = None, **kw: Any):
        """Stream node-level orchestration events (drives the dashboard graph)."""
        sid = session_id or f"flow_{uuid.uuid4().hex[:12]}"
        run = _FlowRun(sid)
        value: Any = input
        try:
            for idx, step in enumerate(self._steps):
                if isinstance(step, _Single):
                    yield NodeStartEvent(id=step.name, name=step.name, kind=_node_kind(step.node))
                    value = await step.execute(run, value)
                    yield NodeEndEvent(id=step.name, name=step.name, output=_as_text(value)[:600])
                elif isinstance(step, _Parallel):
                    for name, node in step.branches:
                        yield NodeStartEvent(id=name, name=name, kind=_node_kind(node))
                    value = await step.execute(run, value)
                    for name, _ in step.branches:
                        yield NodeEndEvent(id=name, name=name,
                                           output=_as_text(run.node_outputs.get(name))[:400])
                elif isinstance(step, _Conditional):
                    branch = step.then if step.predicate(value) else step.otherwise
                    if branch is not None:
                        yield NodeStartEvent(id=branch.name, name=branch.name, kind=_node_kind(branch.node))
                        value = await branch.execute(run, value)
                        yield NodeEndEvent(id=branch.name, name=branch.name, output=_as_text(value)[:600])
                else:  # loop / map
                    name = getattr(step, "name", f"step_{idx}")
                    yield NodeStartEvent(id=name, name=name, kind="function")
                    value = await step.execute(run, value)
                    yield NodeEndEvent(id=name, name=name, output=_as_text(value)[:400])
        except Exception as e:  # noqa: BLE001
            from .errors import ErrorInfo

            yield ErrorEvent(error=ErrorInfo(type="AgentError", message=str(e), where="flow",
                                             cause=type(e).__name__))
            return
        yield DoneEvent(result=RunResult(output=_as_text(value), messages=run.messages,
                                         session_id=sid, status="done", usage=run.usage))

    def run(self, input: Any, *, session_id: str | None = None, **kw: Any) -> RunResult:
        import asyncio

        return asyncio.run(self.arun(input, session_id=session_id, **kw))

    def compile(self) -> "Flow":
        """Return the compiled graph (the Flow itself is the runnable seam)."""
        return self

    def describe(self) -> dict:
        """Topology description for the dashboard orchestration graph."""
        nodes: list[dict] = [{"id": "__start__", "label": "start", "kind": "start"}]
        edges: list[dict] = []
        prev = "__start__"
        for idx, step in enumerate(self._steps):
            if isinstance(step, _Single):
                nodes.append({"id": step.name, "label": step.name, "kind": _node_kind(step.node)})
                edges.append({"src": prev, "dst": step.name})
                prev = step.name
            elif isinstance(step, _Parallel):
                join = f"merge_{idx}"
                for name, node in step.branches:
                    nodes.append({"id": name, "label": name, "kind": _node_kind(node)})
                    edges.append({"src": prev, "dst": name})
                    edges.append({"src": name, "dst": join})
                nodes.append({"id": join, "label": "merge", "kind": "merge"})
                prev = join
            elif isinstance(step, _Conditional):
                cond = f"cond_{idx}"
                nodes.append({"id": cond, "label": "?", "kind": "condition"})
                edges.append({"src": prev, "dst": cond})
                nodes.append({"id": step.then.name, "label": step.then.name, "kind": _node_kind(step.then.node)})
                edges.append({"src": cond, "dst": step.then.name, "label": "yes"})
                if step.otherwise is not None:
                    nodes.append({"id": step.otherwise.name, "label": step.otherwise.name,
                                  "kind": _node_kind(step.otherwise.node)})
                    edges.append({"src": cond, "dst": step.otherwise.name, "label": "no"})
                prev = cond  # branches are leaves
            else:
                name = getattr(step, "name", f"step_{idx}")
                nodes.append({"id": name, "label": name, "kind": "function"})
                edges.append({"src": prev, "dst": name})
                prev = name
        nodes.append({"id": "__end__", "label": "end", "kind": "end"})
        edges.append({"src": prev, "dst": "__end__"})
        return {"kind": "flow", "name": self.name, "nodes": nodes, "edges": edges,
                "tools": [], "mcp_servers": [], "a2a_peers": []}

    # --- serving (reuses the agent runtime; UC-3) ------------------------------

    def asgi_app(self, **kw: Any) -> Any:
        from .runtime.app import create_app

        return create_app(self, **kw)  # type: ignore[arg-type]

    def serve(
        self, host: str = "127.0.0.1", port: int = 8080, *, dashboard_port: int | None = None, **kw: Any
    ) -> None:
        from .runtime.app import serve_target

        serve_target(self, host=host, port=port, dashboard_port=dashboard_port, **kw)

    # --- declarative loading (§12.3) -------------------------------------------

    @classmethod
    def from_spec(cls, spec: FlowSpec) -> "Flow":
        from .spec import resolve_ref

        flow = cls(spec.name)
        nodes = {n.id: resolve_ref(n) for n in spec.nodes}
        # Follow unconditional edges from entry to build the sequence; conditional
        # edges become when(...).end() guards routing to finish.
        order = _linearize(spec)
        for node_id in order:
            flow.step(nodes[node_id], name=node_id)
        return flow

    @classmethod
    def from_yaml(cls, path: str) -> "Flow":
        from .spec import load_flow_spec

        return cls.from_spec(load_flow_spec(path))


def _linearize(spec: FlowSpec) -> list[str]:
    """Order nodes by following unconditional edges from the entry node."""
    succ: dict[str, str] = {}
    for e in spec.edges:
        if e.condition is None and isinstance(e.dst, str) and e.dst != "__end__":
            succ.setdefault(e.src, e.dst)
    order: list[str] = []
    seen: set[str] = set()
    cur: str | None = spec.entry
    while cur and cur not in seen:
        order.append(cur)
        seen.add(cur)
        cur = succ.get(cur)
    return order
