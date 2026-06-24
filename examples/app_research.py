"""Research analyst — ReAct agent with real sources + a sandboxed code interpreter.

A single agent that searches Wikipedia and arXiv (real APIs) and runs Python in a
sandboxed subprocess (the Code Interpreter component) to compute or analyze, with
memory across the session. Demonstrates the gateway (tools), code execution, and
observability components together.

    bash examples/start_research.sh           # or: PORT=8813 python examples/app_research.py
"""

from __future__ import annotations

import os

from agentkit import Agent
from agentkit.memory import MemoryConfig
from agentkit.tools import code_tool
from realtools import arxiv_search, load_dotenv, wikipedia_search, wikipedia_summary

load_dotenv()
GEMINI = f"gemini:{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"

agent = Agent(
    name="research-analyst",
    model=GEMINI,
    instructions=(
        "You are a research analyst. Use wikipedia_search + wikipedia_summary for background and "
        "arxiv_search for recent papers. When a question needs computation, data wrangling, or a "
        "quick simulation, write Python and run it with run_code rather than guessing. Cite sources "
        "(titles/URLs). Be concise."
    ),
    tools=[wikipedia_search, wikipedia_summary, arxiv_search, code_tool(trust="trusted")],
    memory=MemoryConfig(strategies=["semantic", "summary"]),
    observability="langfuse",
)

EXAMPLES = [
    "Summarize the discovery of CRISPR and list 2 recent arXiv papers on it.",
    "Using Wikipedia for the populations, compute the combined population of Japan, Germany, and Brazil with run_code.",
    "What is the Fibonacci sequence? Use run_code to print the first 15 terms.",
]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8813"))
    dash = int(os.environ.get("DASHBOARD_PORT", str(port + 100)))
    agent.serve(
        port=port,
        dashboard_port=dash,
        examples=EXAMPLES,
        description="Research analyst over Wikipedia + arXiv with a sandboxed Python code interpreter.",
    )
