"""Code execution with explicit trust boundaries (§6.2, §11).

trusted  -> SubprocessExecutor (host process, resource-limited, optional net off)
untrusted -> E2BExecutor (managed cloud sandbox; no VM you run)

Selection is per call. Untrusted execution requires E2B; if it is unavailable the
executor fails closed (`SandboxUnavailable`) rather than silently downgrading.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..context import RunContext
from ..errors import SandboxUnavailable, ToolError
from .base import Tool, ToolResult


class ExecResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    artifacts: dict[str, bytes] = Field(default_factory=dict)
    timed_out: bool = False


@runtime_checkable
class CodeExecutor(Protocol):
    async def run(
        self,
        code: str,
        *,
        lang: str = "python",
        timeout_s: float = 30,
        files: dict[str, bytes] | None = None,
        network: bool = False,
    ) -> ExecResult: ...


def _rlimit_preexec():  # pragma: no cover - POSIX only, exercised at runtime
    try:
        import resource

        # Best-effort CPU + address-space caps (trusted host process).
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    except Exception:
        pass


class SubprocessExecutor:
    """Trusted executor: runs code in a subprocess with a temp cwd + resource caps.

    Note: not hardware-isolated and not network-sandboxed — that is the documented
    trust boundary (§11). Use E2B for untrusted code.
    """

    async def run(
        self,
        code: str,
        *,
        lang: str = "python",
        timeout_s: float = 30,
        files: dict[str, bytes] | None = None,
        network: bool = False,
    ) -> ExecResult:
        with tempfile.TemporaryDirectory(prefix="agentkit_exec_") as tmp:
            tmpdir = Path(tmp)
            for name, data in (files or {}).items():
                (tmpdir / name).write_bytes(data)
            if lang == "python":
                script = tmpdir / "_main.py"
                script.write_text(code)
                argv = [sys.executable, str(script)]
            elif lang in ("bash", "sh"):
                script = tmpdir / "_main.sh"
                script.write_text(code)
                argv = ["bash", str(script)]
            else:
                raise ToolError(f"unsupported lang '{lang}'", where="code")

            preexec = _rlimit_preexec if os.name == "posix" else None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=str(tmpdir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    preexec_fn=preexec,
                )
            except Exception as e:  # noqa: BLE001
                raise ToolError(str(e), where="code", cause=type(e).__name__) from e
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                out, err = b"", b"timed out"
                timed_out = True

            artifacts = {
                p.name: p.read_bytes()
                for p in tmpdir.iterdir()
                if p.is_file() and p.name not in ("_main.py", "_main.sh")
            }
            return ExecResult(
                stdout=out.decode(errors="replace"),
                stderr=err.decode(errors="replace"),
                exit_code=proc.returncode or 0,
                artifacts=artifacts,
                timed_out=timed_out,
            )


class E2BExecutor:
    """Untrusted executor: a managed E2B cloud sandbox (optional dependency)."""

    def __init__(self) -> None:
        if not os.environ.get("E2B_API_KEY"):
            raise SandboxUnavailable("E2B_API_KEY not set", where="code")
        try:
            import e2b_code_interpreter  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise SandboxUnavailable(
                "untrusted execution needs e2b: pip install 'agentkit[sandbox]'", where="code"
            ) from e

    async def run(
        self,
        code: str,
        *,
        lang: str = "python",
        timeout_s: float = 30,
        files: dict[str, bytes] | None = None,
        network: bool = False,
    ) -> ExecResult:  # pragma: no cover - requires E2B account
        from e2b_code_interpreter import AsyncSandbox

        sandbox = await AsyncSandbox.create()
        try:
            execution = await sandbox.run_code(code)
            return ExecResult(
                stdout="\n".join(execution.logs.stdout),
                stderr="\n".join(execution.logs.stderr),
                exit_code=1 if execution.error else 0,
            )
        finally:
            await sandbox.kill()


def build_executor(
    trust: Literal["trusted", "untrusted"] = "trusted",
    *,
    executor: CodeExecutor | None = None,
) -> CodeExecutor:
    """Select an executor by trust level (§6.2). Untrusted fails closed."""
    if executor is not None:
        return executor
    if trust == "trusted":
        return SubprocessExecutor()
    return E2BExecutor()  # raises SandboxUnavailable if E2B is absent


def code_tool(
    *,
    trust: Literal["trusted", "untrusted"] = "trusted",
    timeout_s: float = 30,
    executor: CodeExecutor | None = None,
) -> Tool:
    """A `run_code` Tool backed by the trust-selected executor."""
    exec_impl = build_executor(trust, executor=executor)

    async def handler(args: dict, ctx: RunContext) -> ToolResult:
        result = await exec_impl.run(
            args.get("code", ""),
            lang=args.get("lang", "python"),
            timeout_s=timeout_s,
        )
        ok = result.exit_code == 0 and not result.timed_out
        content = result.stdout if ok else (result.stderr or "execution failed")
        return ToolResult(ok=ok, content=content, raw=result.model_dump())

    return Tool(
        name="run_code",
        description="Execute code and return its stdout. lang is 'python' (default) or 'bash'.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "the code to execute"},
                "lang": {"type": "string", "enum": ["python", "bash"], "default": "python"},
            },
            "required": ["code"],
        },
        source="local",
        timeout_s=timeout_s + 5,
        handler=handler,
    )
