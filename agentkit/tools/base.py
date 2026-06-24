"""Tool model, ToolResult, and the @tool decorator (§3.1.4)."""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..context import RunContext
from ..errors import ErrorInfo, ToolError, ToolTimeout
from .schema import build_schema

# A handler adapts an invocation to an underlying capability (local fn / mcp / a2a).
ToolHandler = Callable[[dict, RunContext], Awaitable["ToolResult"]]


class ToolResult(BaseModel):
    ok: bool
    content: Any = None  # JSON-serializable result returned to the model
    error: ErrorInfo | None = None
    latency_ms: float | None = None
    raw: Any = None  # provider raw, never sent to the model


class Tool(BaseModel):
    """An invokable capability (§3.1.4)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict = Field(default_factory=dict)  # JSON Schema (draft 2020-12)
    source: Literal["local", "mcp", "a2a", "agent"] = "local"
    timeout_s: float = 30
    handler: ToolHandler = Field(exclude=True, repr=False)

    async def invoke(self, args: dict, ctx: RunContext) -> ToolResult:
        """Run the tool with a timeout, normalizing failures to ToolResult."""
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(self.handler(args, ctx), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            err = ToolTimeout(
                f"tool '{self.name}' timed out after {self.timeout_s}s", where=self.name
            )
            return ToolResult(ok=False, error=err.info, latency_ms=_ms(start))
        except ToolError as e:
            return ToolResult(ok=False, error=e.info, latency_ms=_ms(start))
        except Exception as e:  # wrap any handler exception as a typed ToolError
            err = ToolError(str(e), where=self.name, cause=type(e).__name__)
            return ToolResult(ok=False, error=err.info, latency_ms=_ms(start))
        result.latency_ms = _ms(start)
        return result


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _make_local_handler(fn: Callable) -> ToolHandler:
    """Wrap a plain callable as a ToolHandler, injecting ctx when accepted."""
    is_async = inspect.iscoroutinefunction(fn)
    sig = inspect.signature(fn)
    wants_ctx = "ctx" in sig.parameters or "context" in sig.parameters
    ctx_key = "ctx" if "ctx" in sig.parameters else "context"

    async def handler(args: dict, ctx: RunContext) -> ToolResult:
        call_kwargs = dict(args)
        if wants_ctx:
            call_kwargs[ctx_key] = ctx
        if is_async:
            value = await fn(**call_kwargs)
        else:
            # CPU-bound / blocking local work runs off the event loop (§3.5).
            value = await asyncio.to_thread(functools.partial(fn, **call_kwargs))
        if isinstance(value, ToolResult):
            return value
        return ToolResult(ok=True, content=value)

    return handler


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    timeout_s: float = 30,
) -> Tool | Callable[[Callable], Tool]:
    """Decorator that wraps a function as a Tool, inferring its JSON Schema.

    Usage::

        @tool
        def add(a: int, b: int) -> int:
            "Add two numbers."
            return a + b

        @tool(name="search", timeout_s=60)
        def search_db(query: str) -> list[str]: ...
    """

    def wrap(func: Callable) -> Tool:
        description, schema = build_schema(func)
        t = Tool(
            name=name or func.__name__,
            description=description,
            parameters=schema,
            source="local",
            timeout_s=timeout_s,
            handler=_make_local_handler(func),
        )
        # Preserve access to the original callable for direct use/testing.
        t.__dict__["__wrapped__"] = func
        return t

    if fn is not None:
        return wrap(fn)
    return wrap
