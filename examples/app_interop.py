"""MCP + A2A interop explorer — one orchestrator consuming a gateway + a peer.

Starts two provider servers in-process:
  * a "toolbox" agent that EXPOSES tools over MCP  (gateway)        :9101/mcp
  * a "researcher" agent served as an A2A peer (+ Agent Card)        :9102/a2a

…then serves an "orchestrator" agent (with its dashboard) that CONSUMES the MCP
tools and the A2A peer as ordinary tools. Run a task and watch the dashboard's
tool timeline show calls routed to the remote MCP tool and the A2A peer.

    bash examples/start_interop.sh            # or: PORT=8814 python examples/app_interop.py
"""

from __future__ import annotations

import os
import threading
import time

import httpx
import uvicorn

from agentkit import Agent
from realtools import fx_rate, load_dotenv, wikipedia_search, wikipedia_summary

load_dotenv()
GEMINI = f"gemini:{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
OPENAI = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}"
MCP_PORT, A2A_PORT = 9101, 9102


def serve_bg(agent: Agent, port: int) -> None:
    app = agent.asgi_app(db_url=f"sqlite:///./_interop_{port}.db")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()


def wait_ready(port: int, timeout: float = 20) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.3)


# Providers
toolbox = Agent(name="toolbox", model=GEMINI, tools=[fx_rate, wikipedia_search, wikipedia_summary],
                instructions="You expose utility tools (FX rates, Wikipedia).")
researcher = Agent(name="researcher", model=GEMINI,
                   instructions="Answer the research question with just the essential fact, concisely.")

# Consumer / orchestrator (gets the dashboard)
orchestrator = Agent(
    name="orchestrator",
    model=OPENAI,
    instructions=(
        "You can use gateway tools provided over MCP (FX rates, Wikipedia) and delegate research "
        "questions to a peer agent over A2A (the 'delegate_*' tool). Pick the right tool for each "
        "sub-task and combine the results."
    ),
    mcp_servers=[f"http://127.0.0.1:{MCP_PORT}/mcp"],
    a2a_peers=[f"http://127.0.0.1:{A2A_PORT}"],
    observability="langfuse",
)

EXAMPLES = [
    "Convert 1000 USD to EUR using the gateway, and ask the peer what year the euro was introduced.",
    "Ask the researcher peer who invented the World Wide Web, then look that person up on Wikipedia.",
    "What's 250 GBP in JPY right now?",
]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8814"))
    print(f"starting MCP gateway :{MCP_PORT} and A2A peer :{A2A_PORT} ...")
    serve_bg(toolbox, MCP_PORT)
    serve_bg(researcher, A2A_PORT)
    wait_ready(MCP_PORT)
    wait_ready(A2A_PORT)
    dash = int(os.environ.get("DASHBOARD_PORT", str(port + 100)))
    orchestrator.serve(
        port=port,
        dashboard_port=dash,
        examples=EXAMPLES,
        description="Orchestrator consuming an MCP gateway + an A2A peer; watch cross-agent calls in the execution trace.",
    )
