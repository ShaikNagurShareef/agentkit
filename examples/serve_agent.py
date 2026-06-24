"""M2: serve an agent over HTTP (the M2 exit criterion).

Run with a provider key set:

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/serve_agent.py

Then, in another shell:

    # non-streaming -> RunResult JSON
    curl -s -X POST localhost:8080/invoke \
      -H 'content-type: application/json' \
      -d '{"input": "What is 1234 * 5678?"}' | jq

    # streaming -> Server-Sent Events (tokens as they arrive)
    curl -N -X POST localhost:8080/invoke \
      -H 'content-type: application/json' \
      -d '{"input": "What is 1234 * 5678?", "stream": true, "session_id": "demo"}'

    # fetch the last run for a session, or resume an interrupted one
    curl -s localhost:8080/runs/demo | jq
    curl -s -X POST localhost:8080/runs/demo/resume | jq

    # ops
    curl -s localhost:8080/healthz
    curl -s localhost:8080/metrics
"""

from __future__ import annotations

from agentkit import Agent, tool


@tool
def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: an arithmetic expression like "12 * (3 + 4)"
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("only basic arithmetic is allowed")
    return float(eval(expression, {"__builtins__": {}}, {}))


agent = Agent(
    name="assistant",
    model="anthropic:claude-opus-4-8",
    instructions="You are a helpful assistant. Use tools when they help.",
    tools=[calculator],
    max_steps=8,
)


if __name__ == "__main__":
    # Binds 127.0.0.1 by default; durable runs persist to ./agentkit.db.
    agent.serve(port=8080)
