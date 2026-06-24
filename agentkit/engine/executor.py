"""Drive a compiled graph to a terminal state and build a RunResult (§3.1.2)."""

from __future__ import annotations

import time
from contextlib import nullcontext as _nullcontext
from typing import TYPE_CHECKING, AsyncIterator

from langgraph.errors import GraphRecursionError

from ..context import RunContext, use_context
from ..errors import AgentError, ErrorInfo, MaxStepsExceeded
from ..types import (
    DoneEvent,
    ErrorEvent,
    Message,
    RunEvent,
    RunResult,
    StepEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
    Usage,
)

if TYPE_CHECKING:
    from ..agent import Agent


class Executor:
    """Runs an Agent's compiled graph for one invocation."""

    def __init__(self, agent: "Agent", graph: object | None = None) -> None:
        self.agent = agent
        # Served app passes a graph compiled against the durable lifespan saver;
        # in-process callers fall back to the agent's cached graph.
        self._graph = graph

    def _get_graph(self):
        return self._graph if self._graph is not None else self.agent.as_graph()

    def _initial_state(self, input: str | Message, session_id: str) -> dict:
        msg = input if isinstance(input, Message) else Message(role="user", content=input)
        return {
            "messages": [msg],
            "session_id": session_id,
            "step": 0,
            "scratchpad": {},
            "pending_tool_calls": [],
            "memory_hits": [],
            "status": "running",
            "error": None,
            "usage": Usage(),
            "tool_records": [],
        }

    def _make_context(
        self, session_id: str, deadline_s: float | None
    ) -> RunContext:
        deadline = time.monotonic() + deadline_s if deadline_s else None
        # Treat each session as a user so per-user (long-term) memory is keyed and
        # populated; callers can override metadata["user_id"] for cross-session users.
        return RunContext(
            session_id=session_id, deadline=deadline, metadata={"user_id": session_id}
        )

    def _config(self, thread_id: str) -> dict:
        # Size the graph recursion limit so our own max_steps check fires first
        # (each step is ~2 supersteps: model + tool), with headroom for the
        # memory_read/memory_write nodes.
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.agent.max_steps * 2 + 10,
        }

    def _result_from_state(self, state: dict, session_id: str) -> RunResult:
        messages: list[Message] = list(state.get("messages", []))
        output = ""
        for m in reversed(messages):
            if m.role == "assistant" and m.content:
                output = m.content
                break
        return RunResult(
            output=output,
            messages=messages,
            tool_calls=list(state.get("tool_records", [])),
            session_id=session_id,
            status="done",
            usage=state.get("usage") or Usage(),
        )

    async def arun(
        self,
        input: str | Message,
        *,
        session_id: str,
        thread_id: str,
        deadline_s: float | None = None,
    ) -> RunResult:
        graph = self._get_graph()
        ctx = self._make_context(session_id, deadline_s)
        config = self._config(thread_id)
        tracer = getattr(self.agent, "_tracer", None)
        trace = tracer.trace(f"run:{session_id}") if tracer else None
        ctx.trace_id = trace.trace_id if trace else None
        cm = trace if trace else _nullcontext()
        with use_context(ctx), cm:
            try:
                final = await graph.ainvoke(self._initial_state(input, session_id), config)
            except GraphRecursionError as e:
                return RunResult(
                    output="",
                    session_id=session_id,
                    status="error",
                    error=MaxStepsExceeded(str(e), where="engine").info,
                    trace_url=trace.trace_url if trace else None,
                )
            except AgentError as e:
                return RunResult(
                    output="", session_id=session_id, status="error", error=e.info,
                    trace_url=trace.trace_url if trace else None,
                )
            except Exception as e:  # NFR-4: never leak a bare exception
                return RunResult(
                    output="",
                    session_id=session_id,
                    status="error",
                    error=ErrorInfo(
                        type="AgentError", message=str(e), where="run", cause=type(e).__name__
                    ),
                    trace_url=trace.trace_url if trace else None,
                )
        result = self._result_from_state(final, session_id)
        result.trace_url = trace.trace_url if trace else None
        return result

    async def astream(
        self,
        input: str | Message,
        *,
        session_id: str,
        thread_id: str,
        deadline_s: float | None = None,
    ) -> AsyncIterator[RunEvent]:
        graph = self._get_graph()
        ctx = self._make_context(session_id, deadline_s)
        ctx.metadata["stream"] = True  # model node streams provider tokens
        config = self._config(thread_id)
        with use_context(ctx):
            try:
                async for mode, chunk in graph.astream(
                    self._initial_state(input, session_id),
                    config,
                    stream_mode=["updates", "custom"],
                ):
                    if mode == "custom":
                        # token chunk emitted by the model node's stream writer
                        if isinstance(chunk, dict) and chunk.get("type") == "token":
                            yield TokenEvent(text=chunk.get("text", ""))
                        continue
                    for node, delta in chunk.items():
                        async for ev in self._events_for_node(node, delta):
                            yield ev
            except GraphRecursionError as e:
                yield ErrorEvent(error=MaxStepsExceeded(str(e), where="engine").info)
                return
            except AgentError as e:
                yield ErrorEvent(error=e.info)
                return
            except Exception as e:  # NFR-4: never leak a bare exception
                yield ErrorEvent(
                    error=ErrorInfo(
                        type="AgentError", message=str(e), where="run", cause=type(e).__name__
                    )
                )
                return
        # Terminal state reload for the final result.
        final = await graph.aget_state(config)
        yield DoneEvent(result=self._result_from_state(final.values, session_id))

    async def _events_for_node(self, node: str, delta: dict) -> AsyncIterator[RunEvent]:
        # Token text arrives via the custom stream (see astream); here we emit the
        # structural events (step boundaries, tool start/end).
        if node == "model":
            yield StepEvent(step=delta.get("step", 0))
            for call in delta.get("pending_tool_calls", []):
                yield ToolStartEvent(id=call.id, name=call.name, args=call.args)
        elif node == "tool":
            for rec in delta.get("tool_records", []):
                yield ToolEndEvent(id=rec.id, name=rec.name, ok=rec.ok, content=rec.content)

    # --- served-app helpers (M2) ----------------------------------------------

    async def get_run(self, *, session_id: str, thread_id: str) -> RunResult | None:
        """Return the last RunResult for a thread from its checkpoint, or None."""
        graph = self._get_graph()
        snapshot = await graph.aget_state(self._config(thread_id))
        if not snapshot or not snapshot.values:
            return None
        result = self._result_from_state(snapshot.values, session_id)
        # status reflects whether the run is mid-flight (pending next node) or done.
        result.status = "interrupted" if snapshot.next else "done"
        return result

    async def aresume(
        self, *, session_id: str, thread_id: str, deadline_s: float | None = None
    ) -> RunResult:
        """Continue an interrupted run from its checkpoint (UC-5)."""
        graph = self._get_graph()
        ctx = self._make_context(session_id, deadline_s)
        config = self._config(thread_id)
        with use_context(ctx):
            try:
                final = await graph.ainvoke(None, config)
            except GraphRecursionError as e:
                return RunResult(
                    output="",
                    session_id=session_id,
                    status="error",
                    error=MaxStepsExceeded(str(e), where="engine").info,
                )
            except AgentError as e:
                return RunResult(
                    output="", session_id=session_id, status="error", error=e.info
                )
            except Exception as e:  # NFR-4
                return RunResult(
                    output="",
                    session_id=session_id,
                    status="error",
                    error=ErrorInfo(
                        type="AgentError", message=str(e), where="resume", cause=type(e).__name__
                    ),
                )
        return self._result_from_state(final, session_id)
