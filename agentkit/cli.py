"""AgentKit CLI (§13.2).

    agentkit run   <target> --input "..." [--session ID] [--stream]
    agentkit serve <target> [--host H] [--port P] [--no-mcp] [--no-a2a]
    agentkit eval  <target> --dataset FILE --metrics task_success,tool_correctness
    agentkit resume <target> <session_id>

``<target>`` is a ``module:attribute`` reference or a ``.yaml`` spec file
(``kind: Agent`` or ``kind: Flow``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_METRICS = {}


def _metric_registry():
    if not _METRICS:
        from .observability import Faithfulness, Latency, TaskSuccess, ToolCorrectness

        _METRICS.update(
            task_success=TaskSuccess,
            tool_correctness=ToolCorrectness,
            faithfulness=Faithfulness,
            latency=Latency,
        )
    return _METRICS


def _load(target: str):
    from .spec import load_target

    return load_target(target)


def _cmd_run(args) -> int:
    target = _load(args.target)
    if args.stream:
        async def _stream():
            async for ev in await target.arun(args.input, session_id=args.session, stream=True):
                if ev.type == "token":
                    print(ev.text, end="", flush=True)
                elif ev.type == "done":
                    print()
                    return ev.result

        result = asyncio.run(_stream())
    else:
        result = target.run(args.input, session_id=args.session)
        print(result.output)
    return 0 if result and result.status == "done" else 1


def _cmd_serve(args) -> int:
    target = _load(args.target)
    target.serve(host=args.host, port=args.port, mcp=not args.no_mcp, a2a=not args.no_a2a)
    return 0


def _cmd_eval(args) -> int:
    from .observability import EvalItem, EvalRunner
    from .spec import _read_doc

    target = _load(args.target)
    raw = _read_doc(args.dataset) if Path(args.dataset).exists() else json.loads(args.dataset)
    items = [EvalItem(**d) for d in raw]
    reg = _metric_registry()
    metrics = [reg[name.strip()]() for name in args.metrics.split(",") if name.strip() in reg]
    runner = EvalRunner(target, dataset=items, metrics=metrics)
    report = asyncio.run(runner.run(sample=args.sample))
    print(json.dumps(report.metrics, indent=2))
    return 0


def _cmd_resume(args) -> int:
    from .engine.compiler import GraphCompiler
    from .engine.executor import Executor
    from .runtime.checkpoint import SqliteCheckpointer

    agent = _load(args.target)

    async def _run():
        saver_cm = SqliteCheckpointer().saver()
        saver = await saver_cm.__aenter__()
        setup = getattr(saver, "setup", None)
        if setup is not None:
            await setup()
        graph = GraphCompiler(agent, agent.provider).compile(checkpointer=saver)
        try:
            return await Executor(agent, graph=graph).aresume(
                session_id=args.session_id, thread_id=args.session_id
            )
        finally:
            await saver_cm.__aexit__(None, None, None)

    result = asyncio.run(_run())
    print(result.output)
    return 0 if result.status == "done" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an agent/flow in-process")
    p_run.add_argument("target")
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--session")
    p_run.add_argument("--stream", action="store_true")

    p_serve = sub.add_parser("serve", help="serve an agent/flow over HTTP")
    p_serve.add_argument("target")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--no-mcp", action="store_true")
    p_serve.add_argument("--no-a2a", action="store_true")

    p_eval = sub.add_parser("eval", help="run a dataset through a target")
    p_eval.add_argument("target")
    p_eval.add_argument("--dataset", required=True, help="JSON/YAML file or inline JSON")
    p_eval.add_argument("--metrics", default="task_success,tool_correctness")
    p_eval.add_argument("--sample", type=int, default=None)

    p_resume = sub.add_parser("resume", help="resume an interrupted run by session id")
    p_resume.add_argument("target")
    p_resume.add_argument("session_id")

    args = parser.parse_args(argv)
    try:
        return {
            "run": _cmd_run,
            "serve": _cmd_serve,
            "eval": _cmd_eval,
            "resume": _cmd_resume,
        }[args.command](args)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
