"""Compile an Agent into a runnable graph (§3.1.2).

The agent compiles to a four-node graph (Figure 2):

    memory_read -> model -> tool -> (loop) -> memory_write -> END

Built on a LangGraph ``StateGraph`` kept behind the ``CompiledGraph`` interface so
the engine backend can be swapped later (§14). In M1 the memory nodes are
pass-through no-ops (memory lands in M4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from langgraph.graph import END, START, StateGraph

from ..context import RunContext, current_context
from ..errors import ModelError
from ..errors import DeadlineExceeded, MaxStepsExceeded
from ..tools.base import Tool
from ..types import Message, ToolCallRecord
from .state import AgentState

if TYPE_CHECKING:
    from ..agent import Agent
    from ..models.base import ModelProvider


class CompiledGraph(Protocol):
    """Engine-agnostic compiled graph the executor drives (§14 swap seam)."""

    async def ainvoke(self, state: dict, config: dict | None = None) -> dict: ...

    def astream(self, state: dict, config: dict | None = None) -> Any: ...


class GraphCompiler:
    """Builds a LangGraph StateGraph from an Agent."""

    def __init__(self, agent: "Agent", provider: "ModelProvider") -> None:
        self.agent = agent
        self.provider = provider
        self.tools: dict[str, Tool] = {t.name: t for t in agent.resolved_tools}

    # --- nodes -----------------------------------------------------------------

    async def _memory_read(self, state: AgentState) -> dict:
        """Scoped vector recall; top-k injected before the model call (§5.3)."""
        store = self.agent._memory_store
        if store is None or not hasattr(store, "recall"):
            return {"memory_hits": []}
        ctx = current_context() or RunContext(session_id=state.get("session_id", ""))
        query = ""
        for m in reversed(state["messages"]):
            if m.role == "user" and m.content:
                query = m.content
                break
        cfg = self.agent._memory_cfg
        k = cfg.top_k if cfg else 6
        with self.agent._tracer.span("memory_read", query=query):
            hits = await store.recall(query, ctx=ctx, k=k)
        return {"memory_hits": hits}

    def _effective_instructions(self, state: AgentState) -> str | None:
        """Augment the system instructions with recalled memory (§5.3)."""
        instructions = self.agent.instructions
        hits = state.get("memory_hits") or []
        if hits:
            block = "\n".join(f"- {h.text}" for h in hits)
            instructions = f"{instructions or ''}\n\nRelevant memory:\n{block}".strip()
        return instructions

    async def _model(self, state: AgentState) -> dict:
        ctx = current_context() or RunContext(session_id=state.get("session_id", ""))
        if ctx.expired():
            raise DeadlineExceeded("run deadline exceeded", where="model")
        step = state.get("step", 0)
        if step >= self.agent.max_steps:
            raise MaxStepsExceeded(
                f"exceeded max_steps={self.agent.max_steps}", where="model"
            )

        instructions = self._effective_instructions(state)
        with self.agent._tracer.generation("model_call", step=step):
            if ctx.metadata.get("stream"):
                resp = await self._stream_model(state, ctx, instructions)
            else:
                resp = await self.provider.complete(
                    list(state["messages"]),
                    tools=list(self.tools.values()),
                    settings=self.agent.model_settings,
                    ctx=ctx,
                    instructions=instructions,
                )
        update: dict = {
            "messages": [resp.message],
            "step": step + 1,
            "usage": resp.usage,
            "pending_tool_calls": resp.tool_calls,
            "status": "awaiting_tool" if resp.tool_calls else "running",
        }
        return update

    async def _stream_model(
        self, state: AgentState, ctx: RunContext, instructions: str | None
    ) -> Any:
        """Stream tokens from the provider to the LangGraph custom stream writer.

        Each text delta is pushed as a ``{"type": "token", "text": ...}`` chunk
        (surfaced to the HTTP layer as a TokenEvent); the terminating delta carries
        the full ModelResponse used to update state.
        """
        from langgraph.config import get_stream_writer

        try:
            writer = get_stream_writer()
        except Exception:  # not inside a custom-stream run
            writer = None

        final = None
        async for delta in self.provider.stream(
            list(state["messages"]),
            tools=list(self.tools.values()),
            settings=self.agent.model_settings,
            ctx=ctx,
            instructions=instructions,
        ):
            if delta.text and writer is not None:
                writer({"type": "token", "text": delta.text})
            if delta.final is not None:
                final = delta.final
        if final is None:
            raise ModelError("stream produced no final response", where="model")
        return final

    async def _tool(self, state: AgentState) -> dict:
        ctx = current_context() or RunContext(session_id=state.get("session_id", ""))
        messages: list[Message] = []
        records: list[ToolCallRecord] = []
        for call in state.get("pending_tool_calls", []):
            tool = self.tools.get(call.name)
            if tool is None:
                content: Any = f"Error: unknown tool '{call.name}'"
                ok = False
                record = ToolCallRecord(
                    id=call.id, name=call.name, args=call.args, ok=False, content=content
                )
            else:
                with self.agent._tracer.span("tool_call", tool=call.name):
                    result = await tool.invoke(call.args, ctx)
                ok = result.ok
                content = (
                    result.content
                    if result.ok
                    else f"Error: {result.error.message if result.error else 'tool failed'}"
                )
                record = ToolCallRecord(
                    id=call.id,
                    name=call.name,
                    args=call.args,
                    ok=result.ok,
                    content=result.content if result.ok else None,
                    error=result.error,
                    latency_ms=result.latency_ms,
                )
                if not result.ok and self.agent.on_tool_error == "raise":
                    from ..errors import ToolError

                    raise ToolError(
                        record.error.message if record.error else "tool failed",
                        where=call.name,
                    )
            messages.append(
                Message(
                    role="tool",
                    name=call.name,
                    tool_call_id=call.id,
                    content=content if isinstance(content, str) else str(content),
                )
            )
            records.append(record)
        return {"messages": messages, "pending_tool_calls": [], "tool_records": records}

    async def _memory_write(self, state: AgentState) -> dict:
        """Extract facts/summaries to long-term memory off the critical path (§5.3)."""
        store = self.agent._memory_store
        if store is None:
            return {"status": "done"}
        from ..memory.strategies import build_summary, extract_preferences, extract_semantic

        ctx = current_context() or RunContext(session_id=state.get("session_id", ""))
        messages = list(state["messages"])
        cfg = self.agent._memory_cfg
        strategies = cfg.strategies if cfg else ["semantic", "summary"]
        has_user = ctx.metadata.get("user_id") is not None

        # The most recent turn (last user + last assistant message) = short-term.
        recent = [m for m in messages if m.role in ("user", "assistant") and m.content][-2:]

        async def _write() -> None:
            with self.agent._tracer.span("memory_write"):
                facts = extract_semantic(messages) if "semantic" in strategies else []
                # session (short-term): just this turn
                if recent:
                    await store.add(recent, scope="session", ctx=ctx)
                # agent (global long-term): distilled facts shared across users
                if facts:
                    await store.add(facts, scope="agent", ctx=ctx)
                # user (per-user long-term): facts + detected preferences
                if has_user:
                    user_items = list(facts)
                    if "user_preference" in strategies:
                        user_items += extract_preferences(messages)
                    if user_items:
                        await store.add(user_items, scope="user", ctx=ctx)
                elif "user_preference" in strategies:
                    await store.add(extract_preferences(messages), scope="user", ctx=ctx)
                if "summary" in strategies and hasattr(store, "set_summary"):
                    store.set_summary(ctx.session_id, build_summary(messages))
                if hasattr(store, "add_transcript"):
                    store.add_transcript(ctx.session_id, messages)

        # write_async would background LLM-based extraction under a long-lived loop;
        # the local heuristic extractors are instant, so we await for determinism.
        await _write()
        return {"status": "done"}

    # --- routing ---------------------------------------------------------------

    def _route_after_model(self, state: AgentState) -> str:
        return "tool" if state.get("pending_tool_calls") else "memory_write"

    # --- build -----------------------------------------------------------------

    def compile(self, checkpointer: Any | None = None) -> CompiledGraph:
        graph = StateGraph(AgentState)
        graph.add_node("memory_read", self._memory_read)
        graph.add_node("model", self._model)
        graph.add_node("tool", self._tool)
        graph.add_node("memory_write", self._memory_write)

        graph.add_edge(START, "memory_read")
        graph.add_edge("memory_read", "model")
        graph.add_conditional_edges(
            "model", self._route_after_model, {"tool": "tool", "memory_write": "memory_write"}
        )
        graph.add_edge("tool", "model")
        graph.add_edge("memory_write", END)

        return graph.compile(checkpointer=checkpointer)
