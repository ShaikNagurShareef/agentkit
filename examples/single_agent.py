"""UC-1: a single agent runs a tool-using task in-process (the M1 exit criterion).

Run with a provider key set, e.g.:

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/single_agent.py

Use a different provider by changing the model string, e.g. "openai:gpt-4o",
"gemini:gemini-2.0-flash", or "groq:llama-3.3-70b-versatile" (install the
matching extra and set that provider's API key).
"""

from __future__ import annotations

import datetime as _dt

from agentkit import Agent, tool


@tool
def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: an arithmetic expression like "12 * (3 + 4)"
    """
    # Restricted eval: arithmetic only.
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("only basic arithmetic is allowed")
    return float(eval(expression, {"__builtins__": {}}, {}))


@tool
def current_date() -> str:
    """Return today's date in ISO format."""
    return _dt.date.today().isoformat()


def main() -> None:
    agent = Agent(
        name="assistant",
        model="anthropic:claude-opus-4-8",
        instructions="You are a helpful assistant. Use tools when they help.",
        tools=[calculator, current_date],
        max_steps=8,
    )

    result = agent.run("What is 1234 * 5678, and what is today's date?")

    print("=== output ===")
    print(result.output)
    print("\n=== tool calls ===")
    for rec in result.tool_calls:
        print(f"  {rec.name}({rec.args}) -> ok={rec.ok} content={rec.content}")
    print(f"\nstatus={result.status} usage={result.usage.total_tokens} tokens")


if __name__ == "__main__":
    main()
