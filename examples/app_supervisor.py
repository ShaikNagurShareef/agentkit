"""Hierarchical / supervisory multi-agent system — a research desk.

A supervisor agent delegates to specialist subagents exposed as tools
(`Agent.as_tool()`). This is the "subagents are treated as tools" pattern from
Anthropic's *Building Effective AI Agents*: the supervisor's ReAct loop calls
ask_researcher / ask_analyst / ask_writer, each a full agent with its own tools.

    bash examples/start_supervisor.sh     # Agent UI :8815 · dashboard :8915

In the dashboard the supervisor's orchestration graph animates, and the event
timeline shows each delegation (purple `agent`-source tool calls).
"""

from __future__ import annotations

import os

from agentkit import Agent
from agentkit.tools import code_tool
from realtools import arxiv_search, load_dotenv, wikipedia_search, wikipedia_summary

load_dotenv()
GEMINI = f"gemini:{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
OPENAI = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}"

# --- specialist subagents -----------------------------------------------------
researcher = Agent(
    name="researcher", model=GEMINI,
    instructions="Find and summarize factual information with sources (titles/URLs) using Wikipedia and arXiv. Be concise.",
    tools=[wikipedia_search, wikipedia_summary, arxiv_search])

analyst = Agent(
    name="analyst", model=GEMINI,
    instructions="Do quantitative analysis and computation. Write and run Python with run_code rather than guessing; report the numbers.",
    tools=[code_tool(trust="trusted")])

writer = Agent(
    name="writer", model=OPENAI,
    instructions="Compose a clear, well-structured final answer from the material you are given. Cite sources where provided.")

# --- supervisor (delegates to the specialists as tools) -----------------------
supervisor = Agent(
    name="research-director",
    model=OPENAI,
    instructions=(
        "You are a research director coordinating a team. Break the request into sub-tasks and "
        "delegate: use ask_researcher to gather facts/sources, ask_analyst for any computation or "
        "data work, and ask_writer to compose the final answer from the gathered material. "
        "Do not answer directly from your own knowledge — orchestrate the specialists, then return "
        "the writer's final answer."
    ),
    tools=[
        researcher.as_tool(name="ask_researcher",
                           description="Delegate fact-finding/literature research (Wikipedia, arXiv)."),
        analyst.as_tool(name="ask_analyst",
                        description="Delegate computation/data analysis (runs Python)."),
        writer.as_tool(name="ask_writer",
                       description="Delegate composing the final written answer from given material."),
    ],
    max_steps=12,
)

EXAMPLES = [
    "Summarize how mRNA vaccines work, find 2 recent arXiv papers, and write a 4-sentence brief.",
    "Find the populations of Japan, Germany, and Brazil, compute their combined total, and write a one-paragraph summary.",
    "Research the history of the transistor and produce a concise timeline with sources.",
]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8815"))
    dash = int(os.environ.get("DASHBOARD_PORT", str(port + 100)))
    supervisor.serve(
        port=port,
        dashboard_port=dash,
        examples=EXAMPLES,
        description="Hierarchical supervisor delegating to researcher / analyst / writer subagents (agent-as-tool).",
    )
