"""Evaluator-optimizer multi-agent workflow — generate ↔ critique ↔ revise.

Two agents in iterative cycles (the evaluator-optimizer pattern): a writer
generates, an editor evaluates and gives feedback, and the writer revises. Built
as a Flow so each pass is a node that animates in the dashboard's orchestration
graph:

    draft → critique-1 → revise → critique-2 → finalize

(`Flow.loop(body, until=...)` also exists for early-stopping cycles; this fixed
pipeline is used so every pass is individually visible.)

    bash examples/start_patterns.sh       # Agent UI :8816 · dashboard :8916
"""

from __future__ import annotations

import os

from agentkit import Agent, Flow
from realtools import load_dotenv

load_dotenv()
GEMINI = f"gemini:{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
OPENAI = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}"

writer = Agent(
    name="writer", model=GEMINI,
    instructions=(
        "You are a writer. If you are given a task, write a first draft. If you are given editor "
        "feedback (it will contain 'REVISE'), produce an improved draft that addresses every point. "
        "Output ONLY the draft text."
    ))

editor = Agent(
    name="editor", model=OPENAI,
    instructions=(
        "You are a strict editor. Evaluate the draft for clarity, accuracy, and concision. "
        "Reply with 'REVISE:' followed by specific, numbered feedback, then a line 'DRAFT:' and the "
        "draft you were given verbatim so the writer can improve it. If it is already excellent, "
        "reply with 'APPROVED:' followed by the final draft."
    ))

# Fixed generate→evaluate→optimize pipeline (visible node-by-node in the dashboard).
flow = (
    Flow("evaluator-optimizer")
    .step(writer, name="draft")
    .step(editor, name="critique-1")
    .step(writer, name="revise")
    .step(editor, name="critique-2")
    .step(writer, name="finalize")
)

EXAMPLES = [
    "Write a crisp 3-sentence product description for a noise-cancelling travel pillow.",
    "Draft a one-paragraph plain-English explanation of how HTTPS keeps data private.",
    "Write a short, vivid opening line for a mystery novel set in a lighthouse.",
]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8816"))
    dash = int(os.environ.get("DASHBOARD_PORT", str(port + 100)))
    flow.serve(
        port=port,
        dashboard_port=dash,
        examples=EXAMPLES,
        description="Evaluator-optimizer: writer ↔ editor iterative refinement (draft → critique → revise → finalize).",
    )
