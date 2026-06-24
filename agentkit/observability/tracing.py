"""Tracing (§7.1): one trace per run, nested spans mirroring the engine.

Instrumentation is implicit — the engine wraps nodes in spans; authors write no
tracing code. Backends: a no-op default (zero overhead), a console tracer for
local visibility, and an optional Langfuse tracer that deep-links
``RunResult.trace_url``. Secrets never enter spans (§11).
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class ObsConfig(BaseModel):
    backend: str = "langfuse"  # langfuse | console | none | otel
    sample_rate: float = 1.0


class Span:
    """A timed span usable as a sync or async context manager."""

    def __init__(self, tracer: "BaseTracer", name: str, kind: str = "span", **attrs: Any) -> None:
        self.tracer = tracer
        self.name = name
        self.kind = kind
        self.attrs = attrs
        self._start = 0.0
        self.error: BaseException | None = None

    def __enter__(self) -> "Span":
        self._start = time.perf_counter()
        return self

    def __exit__(self, et, ev, tb) -> bool:
        dur_ms = (time.perf_counter() - self._start) * 1000
        self.error = ev
        self.tracer._emit(self, dur_ms)
        return False

    async def __aenter__(self) -> "Span":
        return self.__enter__()

    async def __aexit__(self, et, ev, tb) -> bool:
        return self.__exit__(et, ev, tb)


class Trace(Span):
    def __init__(self, tracer: "BaseTracer", name: str, **attrs: Any) -> None:
        super().__init__(tracer, name, kind="trace", **attrs)
        self.trace_id = uuid.uuid4().hex
        self.trace_url: str | None = None


@runtime_checkable
class Tracer(Protocol):
    def trace(self, name: str, **attrs: Any) -> Trace: ...
    def span(self, name: str, **attrs: Any) -> Span: ...
    def generation(self, name: str, **attrs: Any) -> Span: ...
    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None: ...
    def flush(self) -> None: ...


class BaseTracer:
    def trace(self, name: str, **attrs: Any) -> Trace:
        return Trace(self, name, **attrs)

    def span(self, name: str, **attrs: Any) -> Span:
        return Span(self, name, "span", **attrs)

    def generation(self, name: str, **attrs: Any) -> Span:
        return Span(self, name, "generation", **attrs)

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        pass

    def flush(self) -> None:
        pass

    def _emit(self, span: Span, dur_ms: float) -> None:  # noqa: D401
        pass


class NoOpTracer(BaseTracer):
    """Zero-overhead default."""


class ConsoleTracer(BaseTracer):
    """Prints spans to stdout for local debugging."""

    def _emit(self, span: Span, dur_ms: float) -> None:
        status = "error" if span.error else "ok"
        print(f"[trace] {span.kind:10} {span.name:20} {dur_ms:7.1f}ms {status}")

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        print(f"[score] {name}={value} ({comment or ''}) trace={trace_id}")


class _LangfuseSpan(Trace):
    """Wraps a Langfuse v4 observation context manager as a Trace/Span."""

    def __init__(self, tracer: "LangfuseTracer", cm: Any, name: str, is_trace: bool, **attrs: Any):
        super().__init__(tracer, name, **attrs)
        self._cm = cm
        self._is_trace = is_trace

    def __enter__(self) -> "_LangfuseSpan":
        try:
            self._cm.__enter__()
            tid = self.tracer._client.get_current_trace_id()  # type: ignore[attr-defined]
            if tid:
                self.trace_id = tid
                if self._is_trace:
                    self.trace_url = self.tracer._client.get_trace_url(trace_id=tid)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return self

    def __exit__(self, et, ev, tb) -> bool:
        try:
            self._cm.__exit__(et, ev, tb)
        except Exception:  # noqa: BLE001
            pass
        if self._is_trace:
            self.tracer.flush()
        return False


class LangfuseTracer(BaseTracer):
    """Langfuse v4 backend: real nested spans + deep-linked trace_url. Fails soft."""

    def __init__(self) -> None:
        from langfuse import Langfuse  # type: ignore

        # Accept LANGFUSE_HOST or LANGFUSE_BASE_URL (region-specific, e.g. us.cloud.langfuse.com).
        self._host = (
            os.environ.get("LANGFUSE_HOST")
            or os.environ.get("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com"
        )
        os.environ.setdefault("LANGFUSE_HOST", self._host)
        self._client = Langfuse(host=self._host)

    def _observation(self, name: str, as_type: str | None):
        try:
            if as_type:
                return self._client.start_as_current_observation(name=name, as_type=as_type)
            return self._client.start_as_current_observation(name=name)
        except TypeError:
            return self._client.start_as_current_observation(name=name)

    def trace(self, name: str, **attrs: Any) -> Trace:
        return _LangfuseSpan(self, self._observation(name, None), name, is_trace=True, **attrs)

    def span(self, name: str, **attrs: Any) -> Span:
        return _LangfuseSpan(self, self._observation(name, "span"), name, is_trace=False, **attrs)

    def generation(self, name: str, **attrs: Any) -> Span:
        return _LangfuseSpan(self, self._observation(name, "generation"), name, is_trace=False, **attrs)

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        try:
            self._client.create_score(name=name, value=value, trace_id=trace_id, comment=comment)
        except Exception:  # noqa: BLE001
            pass

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:  # noqa: BLE001
            pass


def build_tracer(spec: Any) -> BaseTracer:
    """Resolve an observability spec to a tracer.

    Defaults to no-op. ``"console"`` prints spans. ``"langfuse"`` activates only if
    the package is installed and `LANGFUSE_PUBLIC_KEY` is set; otherwise no-op
    (so the default config never requires infra — NFR-1).
    """
    backend = spec.backend if isinstance(spec, ObsConfig) else (spec or "none")
    backend = os.environ.get("AGENTKIT_OBS", backend)
    if backend == "console":
        return ConsoleTracer()
    key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    if backend == "langfuse" and key and key != "REPLACE_ME":
        try:
            return LangfuseTracer()
        except Exception:  # noqa: BLE001
            return NoOpTracer()
    return NoOpTracer()
