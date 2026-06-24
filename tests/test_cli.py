"""M7 CLI: run + eval dispatch on a keyless function-flow target."""

from __future__ import annotations

from agentkit.cli import main

TARGET = "examples.flow_pipeline:flow"


def test_cli_run(capsys):
    rc = main(["run", TARGET, "--input", "  the  quick   fox "])
    assert rc == 0
    out = capsys.readouterr().out
    assert "The Quick Fox" in out


def test_cli_eval(capsys):
    rc = main(
        ["eval", TARGET, "--dataset", '[{"input": "  a   b ", "expected": "A B"}]',
         "--metrics", "task_success"]
    )
    assert rc == 0
    assert "task_success" in capsys.readouterr().out


def test_cli_unknown_target_errors():
    rc = main(["run", "nope.module:thing", "--input", "x"])
    assert rc == 1
