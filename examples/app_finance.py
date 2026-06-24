"""Finance advisory desk — multi-agent FLOW with real market data.

A flow that orchestrates four agents across two providers:

    profiler ──▶ (credit-analyst ‖ investment-advisor) ──▶ advisor-lead

The two analysts run in parallel (one on Gemini, one on OpenAI), each with real
tools (live FX via Frankfurter, live crypto via CoinGecko) plus deterministic
finance math, and the lead reconciles them. The dashboard animates the
orchestration graph node-by-node as it runs.

    bash examples/start_finance.sh            # or: PORT=8812 python examples/app_finance.py
"""

from __future__ import annotations

import os

from agentkit import Agent, Flow
from realtools import (
    compound_growth,
    crypto_price,
    dti_ratio,
    fx_rate,
    load_dotenv,
    loan_payment,
    risk_allocation,
)

load_dotenv()
GEMINI = f"gemini:{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
OPENAI = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}"
DISC = "You are a financial information assistant, not a licensed advisor; give general information."

profiler = Agent(name="profiler", model=GEMINI,
                 instructions="Summarize the client's finances and goals in one structured paragraph.")
credit_analyst = Agent(
    name="credit-analyst", model=GEMINI,
    instructions=DISC + " Assess borrowing capacity using dti_ratio and loan_payment; if currencies are mentioned, use fx_rate. One-paragraph verdict with the numbers.",
    tools=[dti_ratio, loan_payment, fx_rate])
investment_advisor = Agent(
    name="investment-advisor", model=OPENAI,
    instructions=DISC + " Build an investing plan using risk_allocation and compound_growth; if crypto is mentioned, use crypto_price for a live quote. One-paragraph plan with numbers.",
    tools=[risk_allocation, compound_growth, crypto_price])
advisor_lead = Agent(
    name="advisor-lead", model=OPENAI,
    instructions=DISC + " You are given a credit assessment and an investment plan. Reconcile them into a single prioritized recommendation in 4 short bullets.")

flow = (Flow("financial-advisory")
        .step(profiler)
        .parallel(credit_analyst, investment_advisor)
        .step(advisor_lead))

EXAMPLES = [
    "I'm 35, earn $9,000/month, pay $1,200/month in debt, have $50,000 saved, can invest $1,000/month. Goals: buy a $400k home in 2 years and retire by 60. Moderate risk.",
    "I want a $250k mortgage at 6.5% over 30 years and I'm curious about putting 5% into Bitcoin — thoughts?",
    "I earn 8000 EUR/month but my debts are in USD ($1,500/month). Can I afford a $300k loan?",
]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8812"))
    dash = int(os.environ.get("DASHBOARD_PORT", str(port + 100)))
    flow.serve(
        port=port,
        dashboard_port=dash,
        examples=EXAMPLES,
        description="Multi-agent advisory flow: profiler → parallel(credit ‖ investment) → synthesis, with live FX/crypto.",
    )
