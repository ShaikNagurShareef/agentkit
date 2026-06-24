"""Real-world AgentKit examples — Healthcare and Finance customer agents.

Two industry-representative multi-agent systems built on AgentKit:

  HEALTHCARE — a patient intake & triage assistant
      ReAct clinical assistant (tools: triage scoring, drug-interaction lookup, BMI)
      + intake -> triage -> conditional routing (ER guidance vs care coordination)
      + cross-visit memory (remembers allergies/conditions)

  FINANCE — a customer financial advisory assistant
      ReAct loan officer (tools: DTI, amortized payment)
      + profiler -> parallel(credit analyst || investment advisor) -> synthesis

Tools do real, deterministic computation; the LLMs orchestrate and synthesize.
Cross-provider: Gemini handles structuring/analysis, OpenAI handles
customer-facing synthesis (swap freely — that's the point).

    python examples/industry_agents.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agentkit import Agent, Flow, tool
from agentkit.memory import MemoryConfig


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
GEMINI = f"gemini:{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
OPENAI = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}"

DISCLAIMER_MED = "You are a clinical triage assistant, not a doctor; never give a diagnosis, and always advise contacting a licensed clinician."
DISCLAIMER_FIN = "You are a financial information assistant, not a licensed advisor; frame everything as general information, not personalized advice."


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def _num(x) -> float:
    return float(str(x).replace(",", "").replace("$", "").strip())


# =============================================================================
# HEALTHCARE TOOLS
# =============================================================================

_RED_FLAGS = {
    "chest pain": 5, "shortness of breath": 5, "difficulty breathing": 5,
    "stroke": 5, "slurred speech": 5, "numbness": 4, "severe bleeding": 5,
    "fainting": 4, "confusion": 4, "high fever": 3, "persistent vomiting": 3,
    "severe headache": 4, "vision loss": 4,
}

_INTERACTIONS = {
    frozenset({"warfarin", "aspirin"}): ("major", "increased bleeding risk"),
    frozenset({"warfarin", "ibuprofen"}): ("major", "increased bleeding risk"),
    frozenset({"lisinopril", "potassium"}): ("moderate", "risk of hyperkalemia"),
    frozenset({"metformin", "alcohol"}): ("moderate", "risk of lactic acidosis"),
    frozenset({"simvastatin", "grapefruit"}): ("moderate", "elevated statin levels"),
    frozenset({"ssri", "tramadol"}): ("major", "serotonin syndrome risk"),
}


@tool
def triage_score(symptoms: str, age: int = 40) -> dict:
    """Score symptom urgency from a free-text description and patient age.

    Args:
        symptoms: free-text description of what the patient reports
        age: patient age in years
    """
    s = symptoms.lower()
    flags = [k for k in _RED_FLAGS if k in s]
    score = sum(_RED_FLAGS[k] for k in flags)
    if int(age) >= 65:
        score += 1
    level = "emergency" if score >= 5 else "urgent" if score >= 3 else "routine"
    return {"urgency": level, "score": score, "red_flags": flags}


@tool
def drug_interaction(drug_a: str, drug_b: str) -> dict:
    """Check for a known interaction between two medications.

    Args:
        drug_a: first medication name
        drug_b: second medication name
    """
    key = frozenset({drug_a.lower().strip(), drug_b.lower().strip()})
    if key in _INTERACTIONS:
        severity, note = _INTERACTIONS[key]
        return {"interaction": True, "severity": severity, "note": note}
    return {"interaction": False, "severity": "none", "note": "no known interaction in database"}


@tool
def bmi(weight_kg: float, height_cm: float) -> dict:
    """Compute Body Mass Index and category.

    Args:
        weight_kg: weight in kilograms
        height_cm: height in centimeters
    """
    h = _num(height_cm) / 100.0
    value = round(_num(weight_kg) / (h * h), 1)
    cat = ("underweight" if value < 18.5 else "normal" if value < 25
           else "overweight" if value < 30 else "obese")
    return {"bmi": value, "category": cat}


@tool
def find_appointment(specialty: str) -> str:
    """Return the next available appointment slot for a specialty (mock scheduler).

    Args:
        specialty: e.g. 'cardiology', 'primary care'
    """
    slots = {"cardiology": "Tue 9:40 AM", "primary care": "tomorrow 2:15 PM",
             "neurology": "Thu 11:00 AM", "dermatology": "next Mon 3:30 PM"}
    return slots.get(specialty.lower(), "next available: tomorrow 4:00 PM")


# =============================================================================
# FINANCE TOOLS
# =============================================================================

@tool
def dti_ratio(monthly_debt: float, monthly_income: float) -> dict:
    """Compute debt-to-income ratio and lending eligibility band.

    Args:
        monthly_debt: total monthly debt payments
        monthly_income: gross monthly income
    """
    income = _num(monthly_income)
    ratio = round(_num(monthly_debt) / income * 100, 1) if income else 100.0
    band = ("excellent" if ratio <= 28 else "acceptable" if ratio <= 36
            else "elevated" if ratio <= 43 else "high-risk")
    return {"dti_pct": ratio, "band": band, "qualifies_conventional": ratio <= 43}


@tool
def loan_payment(principal: float, annual_rate_pct: float, years: int) -> dict:
    """Compute the fixed monthly payment for an amortized loan.

    Args:
        principal: loan amount
        annual_rate_pct: annual interest rate as a percent (e.g. 6.5)
        years: term in years
    """
    p = _num(principal)
    r = _num(annual_rate_pct) / 100 / 12
    n = int(years) * 12
    payment = p * r / (1 - (1 + r) ** -n) if r else p / n
    return {"monthly_payment": round(payment, 2), "total_paid": round(payment * n, 2),
            "total_interest": round(payment * n - p, 2)}


@tool
def compound_growth(principal: float, annual_rate_pct: float, years: int, contributions_monthly: float = 0.0) -> dict:
    """Project investment growth with optional monthly contributions.

    Args:
        principal: starting amount
        annual_rate_pct: expected annual return percent
        years: number of years
        contributions_monthly: amount added each month
    """
    p = _num(principal)
    r = _num(annual_rate_pct) / 100 / 12
    n = int(years) * 12
    c = _num(contributions_monthly)
    fv = p * (1 + r) ** n + (c * (((1 + r) ** n - 1) / r) if r else c * n)
    return {"future_value": round(fv, 2), "contributed": round(p + c * n, 2)}


@tool
def risk_allocation(age: int, risk_tolerance: str = "moderate") -> dict:
    """Suggest a stock/bond split from age and risk tolerance (rule-of-thumb).

    Args:
        age: investor age
        risk_tolerance: 'conservative' | 'moderate' | 'aggressive'
    """
    base_equity = max(20, min(90, 110 - int(age)))
    adj = {"conservative": -15, "moderate": 0, "aggressive": 15}.get(risk_tolerance.lower(), 0)
    equity = max(10, min(95, base_equity + adj))
    return {"stocks_pct": equity, "bonds_pct": 100 - equity, "tolerance": risk_tolerance}


# =============================================================================
# HEALTHCARE SYSTEM
# =============================================================================

async def healthcare() -> None:
    banner("HEALTHCARE — clinical triage assistant (ReAct)")
    clinician = Agent(
        name="clinical-assistant", model=GEMINI,
        instructions=DISCLAIMER_MED + " Use the tools to assess. Be concise and structured.",
        tools=[triage_score, drug_interaction, bmi, find_appointment], max_steps=10,
    )
    r = await clinician.arun(
        "A 68-year-old reports chest pain and shortness of breath. They take warfarin and were "
        "just prescribed aspirin. Are they at risk, and how urgent is this?"
    )
    print(r.output)
    print("\n[tools]", [f"{c.name}->{c.content}" for c in r.tool_calls])

    banner("HEALTHCARE — intake → triage → conditional routing (multi-agent)")
    intake = Agent(name="intake", model=GEMINI,
                   instructions="Extract a structured one-paragraph clinical summary (age, symptoms, meds) from the patient message.")
    triage = Agent(name="triage", model=GEMINI,
                   instructions=DISCLAIMER_MED + " Call triage_score and drug_interaction on the summary, then state the urgency level (emergency/urgent/routine) clearly in your first line.",
                   tools=[triage_score, drug_interaction])
    er_guidance = Agent(name="er-guidance", model=OPENAI,
                        instructions="This is an emergency triage. Give clear, calm, immediate next-step guidance (call 911 / go to ER) in 3 short bullets.")
    care_coord = Agent(name="care-coordinator", model=OPENAI,
                       instructions="This is non-emergency. Recommend a routine care plan and suggest booking with find_appointment. Keep it reassuring and brief.",
                       tools=[find_appointment])
    flow = (Flow("patient-triage")
            .step(intake).step(triage)
            .when(lambda t: "emergency" in t.lower() or "urgent" in t.lower())
            .then(er_guidance).otherwise(care_coord))

    for case in [
        "I'm 71 and suddenly have severe chest pain and slurred speech.",
        "I'm 29 with a mild sore throat and a runny nose for two days.",
    ]:
        res = await flow.arun(case)
        print(f"\nPATIENT: {case}\nSYSTEM : {res.output}")

    banner("HEALTHCARE — memory across visits (allergy recall)")
    patient_agent = Agent(
        name="patient-portal", model=GEMINI,
        instructions=DISCLAIMER_MED + " Use what you remember about the patient; flag any conflict with their record.",
        memory=MemoryConfig(strategies=["semantic", "user_preference", "summary"]),
    )
    await patient_agent.arun("Please note for my record: I am allergic to penicillin.", session_id="patient-1001")
    r = await patient_agent.arun("My doctor suggested amoxicillin for a sinus infection — any concern for me?", session_id="patient-1001")
    print(r.output)


# =============================================================================
# FINANCE SYSTEM
# =============================================================================

async def finance() -> None:
    banner("FINANCE — loan officer (ReAct)")
    loan_officer = Agent(
        name="loan-officer", model=GEMINI,
        instructions=DISCLAIMER_FIN + " Use the tools to compute. Give a clear qualify/decline read with the numbers.",
        tools=[dti_ratio, loan_payment], max_steps=10,
    )
    r = await loan_officer.arun(
        "I earn $9,000/month and pay $1,200/month in existing debt. I want a $400,000 mortgage at "
        "6.5% over 30 years. What's the monthly payment and do I qualify?"
    )
    print(r.output)
    print("\n[tools]", [f"{c.name}->{c.content}" for c in r.tool_calls])

    banner("FINANCE — profiler → parallel(credit || investment) → synthesis (multi-agent)")
    profiler = Agent(name="profiler", model=GEMINI,
                     instructions="Summarize the client's finances and goals in one structured paragraph.")
    credit_analyst = Agent(name="credit-analyst", model=GEMINI,
                           instructions=DISCLAIMER_FIN + " Assess borrowing capacity. Use dti_ratio and loan_payment. One-paragraph verdict with numbers.",
                           tools=[dti_ratio, loan_payment])
    investment_advisor = Agent(name="investment-advisor", model=OPENAI,
                               instructions=DISCLAIMER_FIN + " Assess investing. Use risk_allocation and compound_growth. One-paragraph plan with numbers.",
                               tools=[risk_allocation, compound_growth])
    synthesizer = Agent(name="advisor-lead", model=OPENAI,
                        instructions=DISCLAIMER_FIN + " You are given a credit assessment and an investment plan. Reconcile them into a single prioritized recommendation for the client in 4 short bullets.")

    flow = (Flow("financial-advisory")
            .step(profiler)
            .parallel(credit_analyst, investment_advisor)
            .step(synthesizer))
    client = (
        "I'm 35, earn $9,000/month, pay $1,200/month in debt, have $50,000 saved, and can invest "
        "$1,000/month. Goals: buy a $400k home in 2 years and retire comfortably by 60. "
        "Moderate risk tolerance."
    )
    res = await flow.arun(client)
    print(f"\nCLIENT: {client}\n\nADVISORY TEAM RECOMMENDATION:\n{res.output}")
    print(f"\n[orchestration: profiler→(credit‖investment)→synthesis · {res.usage.total_tokens} tokens · 2 providers]")


async def main() -> None:
    missing = [k for k in ("GEMINI_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"Missing keys: {', '.join(missing)}. Add them to .env.")
        return
    print(f"Models: structuring/analysis = {GEMINI} · customer-facing synthesis = {OPENAI}")
    await healthcare()
    await finance()
    print(f"\n{'=' * 72}\n  Done — Healthcare & Finance agents built and orchestrated on AgentKit.\n{'=' * 72}")


if __name__ == "__main__":
    asyncio.run(main())
