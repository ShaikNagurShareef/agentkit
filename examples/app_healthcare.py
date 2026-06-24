"""Healthcare triage assistant — ReAct agent + real medical APIs + memory + tracing.

A single reasoning agent that uses live medical data sources (NLM Clinical Tables,
RxNorm/ICD-10, openFDA drug labels) plus a deterministic triage rubric, remembers
patient facts across visits, and is fully traced.

    bash examples/start_healthcare.sh         # or: PORT=8811 python examples/app_healthcare.py

Open the dashboard to explore: live orchestration graph, tool I/O, the Memory
explorer (session/user/agent scopes), Components map, Evals, and traces.
"""

from __future__ import annotations

import os

from agentkit import Agent
from agentkit.memory import MemoryConfig
from realtools import bmi, condition_lookup, drug_label, icd10_lookup, load_dotenv, triage_score

load_dotenv()
GEMINI = f"gemini:{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"

agent = Agent(
    name="triage-assistant",
    model=GEMINI,
    instructions=(
        "You are a clinical triage assistant, not a doctor. Never diagnose; always advise "
        "contacting a licensed clinician. Use the tools: triage_score for urgency, "
        "condition_lookup / icd10_lookup for standardized terms/codes, and drug_label for real "
        "FDA warnings and interactions. Use anything you remember about the patient (allergies, "
        "conditions) and flag conflicts. Be concise and structured."
    ),
    tools=[triage_score, condition_lookup, icd10_lookup, drug_label, bmi],
    memory=MemoryConfig(strategies=["semantic", "user_preference", "summary"]),
    observability="langfuse",
)

EXAMPLES = [
    "I'm 68 with sudden chest pain and shortness of breath; I take warfarin and was just prescribed aspirin. How urgent is this?",
    "Note for my record: I am allergic to penicillin.",
    "My doctor suggested amoxicillin for a sinus infection — any concern for me?",
    "What are the FDA warnings for metformin, and the ICD-10 code for type 2 diabetes?",
]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8811"))
    dash = int(os.environ.get("DASHBOARD_PORT", str(port + 100)))
    agent.serve(
        port=port,
        dashboard_port=dash,
        examples=EXAMPLES,
        description="Clinical triage assistant over real medical data sources with cross-visit memory.",
    )
