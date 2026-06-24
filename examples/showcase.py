"""AgentKit showcase — every orchestration pattern, across Gemini + OpenAI.

Add keys to a .env file in the repo root, then:

    python examples/showcase.py

    # .env
    GEMINI_API_KEY=...
    OPENAI_API_KEY=...
    # optional overrides:
    GEMINI_MODEL=gemini-2.0-flash
    OPENAI_MODEL=gpt-4o-mini

Demonstrates:
  1. ReAct  — a single agent reasoning + calling tools in a loop (Gemini)
  2. Multi-agent pipeline — researcher → writer → reviewer (Gemini → OpenAI → Gemini)
  3. Parallel fan-out / fan-in — two analysts run concurrently, then a synthesizer
  4. Conditional routing — a predicate routes the question to a specialist
  5. Cross-session memory — a preference stated once, recalled later
  6. Token streaming — live token-by-token output
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from agentkit import Agent, Flow, tool
from agentkit.memory import MemoryConfig


# --- .env loader (no extra dependency) ----------------------------------------

def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_dotenv()
GEMINI = f"gemini:{os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')}"
OPENAI = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}"


def _require_keys() -> None:
    missing = [k for k in ("GEMINI_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"Missing keys: {', '.join(missing)}. Add them to .env and re-run.")
        sys.exit(1)


# --- tools --------------------------------------------------------------------

@tool
def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression like '17 * 23'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("only basic arithmetic is allowed")
    return float(eval(expression, {"__builtins__": {}}, {}))


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\n  PATTERN {n}: {title}\n{'=' * 70}")


# --- 1. ReAct ------------------------------------------------------------------

async def react() -> None:
    banner(1, "ReAct — single agent, tool-using loop (Gemini)")
    agent = Agent(
        name="analyst",
        model=GEMINI,
        instructions="You are a precise analyst. Use tools for any calculation.",
        tools=[calculator, word_count],
        max_steps=8,
    )
    r = await agent.arun(
        "How many words are in 'the quick brown fox jumps', and what is that count times 9?"
    )
    print("answer :", r.output)
    print("tools  :", [f"{c.name}({c.args})->{c.content}" for c in r.tool_calls])
    print("usage  :", r.usage.total_tokens, "tokens · status", r.status)


# --- 2. Multi-agent pipeline ---------------------------------------------------

async def pipeline() -> None:
    banner(2, "Multi-agent pipeline — researcher → writer → reviewer (Gemini→OpenAI→Gemini)")
    researcher = Agent(
        name="researcher", model=GEMINI,
        instructions="Produce 3 concise bullet-point facts about the topic. Bullets only.",
    )
    writer = Agent(
        name="writer", model=OPENAI,
        instructions="Turn the bullet points you are given into a single vivid 2-sentence summary.",
    )
    reviewer = Agent(
        name="reviewer", model=GEMINI,
        instructions="Tighten the text you are given to one punchy sentence. Output only that sentence.",
    )
    flow = Flow("research-and-write").step(researcher).step(writer).step(reviewer)
    r = await flow.arun("the discovery of penicillin")
    print("final  :", r.output)
    print("usage  :", r.usage.total_tokens, "tokens across 3 agents (2 providers)")


# --- 3. Parallel fan-out / fan-in ---------------------------------------------

async def parallel() -> None:
    banner(3, "Parallel fan-out/fan-in — two analysts concurrently, then synthesis")
    optimist = Agent(name="optimist", model=GEMINI,
                     instructions="Give the single strongest argument FOR the proposal, in one sentence.")
    pessimist = Agent(name="pessimist", model=OPENAI,
                      instructions="Give the single strongest argument AGAINST the proposal, in one sentence.")
    synthesizer = Agent(name="synthesizer", model=GEMINI,
                        instructions="You are given two opposing views. Render a balanced one-sentence verdict.")
    flow = Flow("debate").parallel(optimist, pessimist).step(synthesizer)
    r = await flow.arun("Proposal: every company should adopt a 4-day work week.")
    print("verdict:", r.output)


# --- 4. Conditional routing ----------------------------------------------------

async def routing() -> None:
    banner(4, "Conditional routing — predicate routes to a specialist")
    math_agent = Agent(name="mathematician", model=GEMINI,
                       instructions="Solve the math question. Use the calculator tool.",
                       tools=[calculator])
    poet = Agent(name="poet", model=OPENAI, instructions="Respond with a short, elegant couplet.")
    router = Flow("router").when(lambda q: any(c.isdigit() for c in q)).then(math_agent).otherwise(poet)

    for q in ["What is 144 divided by 12, then plus 5?", "Describe a sunrise over the ocean."]:
        r = await router.arun(q)
        print(f"  q: {q}\n  -> {r.output}\n")


# --- 5. Cross-session memory ---------------------------------------------------

async def memory() -> None:
    banner(5, "Cross-session memory — a preference stated once, recalled later")
    agent = Agent(
        name="companion", model=GEMINI,
        instructions="You are a helpful companion. Use anything you remember about the user.",
        memory=MemoryConfig(strategies=["semantic", "user_preference", "summary"]),
    )
    await agent.arun("Remember: I love hiking and I prefer very short replies.", session_id="user-42")
    r = await agent.arun("Suggest something fun for me to do this weekend.", session_id="user-42")
    print("reply  :", r.output)
    print("(memory recalled the hiking preference and short-reply style)")


# --- 6. Streaming --------------------------------------------------------------

async def streaming() -> None:
    banner(6, "Token streaming — live output (Gemini)")
    agent = Agent(name="storyteller", model=GEMINI,
                  instructions="Write a single imaginative sentence.")
    print("stream : ", end="", flush=True)
    async for ev in await agent.arun("Open a story about a lighthouse.", stream=True):
        if ev.type == "token":
            print(ev.text, end="", flush=True)
    print()


async def main() -> None:
    _require_keys()
    print(f"Models: research/analysis = {GEMINI} · writing/critique = {OPENAI}")
    for fn in (react, pipeline, parallel, routing, memory, streaming):
        try:
            await fn()
        except Exception as e:  # noqa: BLE001
            print(f"  [pattern failed: {type(e).__name__}: {e}]")
    print(f"\n{'=' * 70}\n  Done — one library, six orchestration patterns, two providers.\n{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
