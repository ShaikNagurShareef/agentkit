"""AgentKit MCP + A2A demo — real servers over HTTP, then consumed as tools.

Add GEMINI_API_KEY and OPENAI_API_KEY to .env, then:

    python examples/protocols_demo.py

What it does:
  * Serves a "toolbox" agent that EXPOSES a calculator tool over **MCP** (/mcp).
  * Serves a "researcher" agent as an **A2A** peer (/a2a + Agent Card).
  * Talks to both directly with MCPClient / A2AClient (protocol-level proof).
  * Builds a "planner" agent that CONSUMES the MCP tool and the A2A peer as
    ordinary tools, and orchestrates them to answer one question.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from agentkit import Agent, tool
from agentkit.protocols import A2AClient, MCPClient
from agentkit.protocols.a2a import artifact_text


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()
GEMINI = f"gemini:{os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')}"
OPENAI = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}"
_DB = Path(tempfile.mkdtemp(prefix="agentkit_proto_"))


@tool
def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression like '2025 - 1928'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("only basic arithmetic is allowed")
    return float(eval(expression, {"__builtins__": {}}, {}))


def serve_in_thread(agent: Agent, port: int) -> uvicorn.Server:
    app = agent.asgi_app(db_url=f"sqlite:///{_DB}/{agent.name}.db")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


def wait_ready(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"server on :{port} did not become ready")


async def direct_mcp() -> None:
    print("\n=== MCP (direct client) ===")
    mc = MCPClient("http://127.0.0.1:9001/mcp", namespace="toolbox")
    await mc.connect()
    print("discovered tools:", [d.name for d in await mc.list_tools()])
    from agentkit.context import RunContext

    res = await mc.call_tool("calculator", {"expression": "2025 - 1928"}, ctx=RunContext(session_id="x"))
    print("calculator(2025-1928) via MCP ->", res.content)
    await mc.close()


async def direct_a2a() -> None:
    print("\n=== A2A (direct client) ===")
    client = A2AClient()
    card = await client.get_card("http://127.0.0.1:9002")
    print("agent card:", card.name, "—", (card.description or "")[:60])
    task = await client.send("http://127.0.0.1:9002", "In what year was penicillin discovered? One number.")
    print("A2A task state:", task["status"]["state"], "| artifact:", artifact_text(task)[:80])


async def integrated() -> None:
    print("\n=== Integrated: planner CONSUMES MCP tool + A2A peer ===")
    planner = Agent(
        name="planner",
        model=OPENAI,
        instructions=(
            "You have a calculator tool (via MCP) and can delegate research to a peer "
            "agent (via A2A). Use the peer to find facts and the calculator to compute. "
            "Show the final number."
        ),
        mcp_servers=["http://127.0.0.1:9001/mcp"],
        a2a_peers=["http://127.0.0.1:9002"],
        max_steps=10,
    )
    r = await planner.arun(
        "Find the year penicillin was discovered, then compute how many years ago that was from 2025."
    )
    print("planner answer :", r.output)
    print("tools used     :", [c.name for c in r.tool_calls])


async def main() -> None:
    missing = [k for k in ("GEMINI_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"Missing keys: {', '.join(missing)}. Add them to .env and re-run.")
        sys.exit(1)

    toolbox = Agent(name="toolbox", model=GEMINI, tools=[calculator],
                    instructions="You expose a calculator.")
    researcher = Agent(name="researcher", model=GEMINI,
                       instructions="Answer the research question with just the essential fact.")

    print("starting MCP server :9001 and A2A server :9002 ...")
    serve_in_thread(toolbox, 9001)   # exposes calculator over /mcp
    serve_in_thread(researcher, 9002)  # exposes /a2a + agent card
    wait_ready(9001)
    wait_ready(9002)

    await direct_mcp()
    await direct_a2a()
    await integrated()
    print("\nDone — MCP and A2A exercised both directly and as orchestrated agent tools.")


if __name__ == "__main__":
    asyncio.run(main())
