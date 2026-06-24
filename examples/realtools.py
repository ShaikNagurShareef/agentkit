"""Real external-API tools for the example apps — free, no API key required.

Healthcare : NLM Clinical Tables (conditions/ICD-10), RxNorm, openFDA drug labels
Finance    : Frankfurter FX rates, CoinGecko crypto prices  (+ deterministic math)
Research   : Wikipedia, arXiv, Hacker News

All calls are real HTTP to public endpoints. Tools return data (or a short error
string) rather than raising, so the agent can recover (on_tool_error="surface").
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from agentkit import tool

UA = {"User-Agent": "AgentKit-demo/1.0"}


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            if v and v != "REPLACE_ME":
                os.environ.setdefault(k.strip(), v)


async def _get_json(url: str, params: dict | None = None, timeout: float = 12):
    async with httpx.AsyncClient(timeout=timeout, headers=UA) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()


def _num(x) -> float:
    return float(str(x).replace(",", "").replace("$", "").strip())


# =========================== HEALTHCARE (real) ===============================

@tool
async def condition_lookup(query: str) -> list:
    """Search standardized medical conditions (NLM Clinical Tables).

    Args:
        query: a symptom or condition term, e.g. 'chest pain'
    """
    try:
        data = await _get_json(
            "https://clinicaltables.nlm.nih.gov/api/conditions/v3/search",
            {"terms": query, "maxList": 6},
        )
        return data[3] if len(data) > 3 else data[1]
    except Exception as e:  # noqa: BLE001
        return [f"lookup error: {e}"]


@tool
async def icd10_lookup(query: str) -> list:
    """Look up ICD-10-CM diagnosis codes for a term (NLM Clinical Tables).

    Args:
        query: a diagnosis term, e.g. 'type 2 diabetes'
    """
    try:
        data = await _get_json(
            "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
            {"terms": query, "maxList": 6},
        )
        return data[3] if len(data) > 3 else []
    except Exception as e:  # noqa: BLE001
        return [f"icd10 error: {e}"]


@tool
async def drug_label(drug: str) -> dict:
    """Fetch real FDA drug-label warnings/interactions for a medication (openFDA).

    Args:
        drug: a generic or brand drug name, e.g. 'warfarin'
    """
    try:
        data = await _get_json(
            "https://api.fda.gov/drug/label.json",
            {"search": f'openfda.generic_name:"{drug}" OR openfda.brand_name:"{drug}"', "limit": 1},
        )
        res = data.get("results", [{}])[0]
        def first(k):
            v = res.get(k)
            return (v[0][:600] if isinstance(v, list) and v else None)
        return {
            "drug": drug,
            "warnings": first("warnings") or first("boxed_warning"),
            "drug_interactions": first("drug_interactions"),
            "indications": first("indications_and_usage"),
        }
    except Exception as e:  # noqa: BLE001
        return {"drug": drug, "error": f"no FDA label found ({e})"}


@tool
def triage_score(symptoms: str, age: int = 40) -> dict:
    """Score symptom urgency (deterministic red-flag rubric).

    Args:
        symptoms: free-text description of the patient's symptoms
        age: patient age in years
    """
    flags = {"chest pain": 5, "shortness of breath": 5, "difficulty breathing": 5,
             "stroke": 5, "slurred speech": 5, "numbness": 4, "severe bleeding": 5,
             "fainting": 4, "confusion": 4, "high fever": 3, "severe headache": 4,
             "vision loss": 4, "persistent vomiting": 3}
    s = symptoms.lower()
    hit = [k for k in flags if k in s]
    score = sum(flags[k] for k in hit) + (1 if int(age) >= 65 else 0)
    level = "emergency" if score >= 5 else "urgent" if score >= 3 else "routine"
    return {"urgency": level, "score": score, "red_flags": hit}


@tool
def bmi(weight_kg: float, height_cm: float) -> dict:
    """Compute BMI and category.

    Args:
        weight_kg: weight in kilograms
        height_cm: height in centimeters
    """
    h = _num(height_cm) / 100
    v = round(_num(weight_kg) / (h * h), 1)
    cat = "underweight" if v < 18.5 else "normal" if v < 25 else "overweight" if v < 30 else "obese"
    return {"bmi": v, "category": cat}


# =========================== FINANCE (real + math) ===========================

@tool
async def fx_rate(amount: float, base: str, quote: str) -> dict:
    """Convert currency at the live ECB reference rate (Frankfurter, real).

    Args:
        amount: amount to convert
        base: source currency code, e.g. 'USD'
        quote: target currency code, e.g. 'EUR'
    """
    try:
        data = await _get_json("https://api.frankfurter.app/latest",
                               {"amount": _num(amount), "from": base.upper(), "to": quote.upper()})
        return {"amount": amount, "base": base.upper(), "quote": quote.upper(),
                "converted": data["rates"].get(quote.upper()), "date": data.get("date")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"fx error: {e}"}


@tool
async def crypto_price(coin: str) -> dict:
    """Get a live cryptocurrency price in USD (CoinGecko, real).

    Args:
        coin: coin id or symbol, e.g. 'bitcoin', 'ethereum', 'btc'
    """
    ids = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana", "ada": "cardano"}
    cid = ids.get(coin.lower(), coin.lower())
    try:
        data = await _get_json("https://api.coingecko.com/api/v3/simple/price",
                               {"ids": cid, "vs_currencies": "usd", "include_24hr_change": "true"})
        if cid not in data:
            return {"error": f"unknown coin '{coin}'"}
        return {"coin": cid, "usd": data[cid]["usd"], "change_24h_pct": round(data[cid].get("usd_24h_change", 0), 2)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"crypto error: {e}"}


@tool
def dti_ratio(monthly_debt: float, monthly_income: float) -> dict:
    """Debt-to-income ratio and lending band.

    Args:
        monthly_debt: total monthly debt payments
        monthly_income: gross monthly income
    """
    inc = _num(monthly_income)
    ratio = round(_num(monthly_debt) / inc * 100, 1) if inc else 100.0
    band = "excellent" if ratio <= 28 else "acceptable" if ratio <= 36 else "elevated" if ratio <= 43 else "high-risk"
    return {"dti_pct": ratio, "band": band, "qualifies_conventional": ratio <= 43}


@tool
def loan_payment(principal: float, annual_rate_pct: float, years: int) -> dict:
    """Fixed monthly payment for an amortized loan.

    Args:
        principal: loan amount
        annual_rate_pct: annual interest rate percent
        years: term in years
    """
    p = _num(principal); r = _num(annual_rate_pct) / 100 / 12; n = int(years) * 12
    pay = p * r / (1 - (1 + r) ** -n) if r else p / n
    return {"monthly_payment": round(pay, 2), "total_interest": round(pay * n - p, 2)}


@tool
def compound_growth(principal: float, annual_rate_pct: float, years: int, contributions_monthly: float = 0.0) -> dict:
    """Project investment growth with optional monthly contributions.

    Args:
        principal: starting amount
        annual_rate_pct: expected annual return percent
        years: number of years
        contributions_monthly: amount added each month
    """
    p = _num(principal); r = _num(annual_rate_pct) / 100 / 12; n = int(years) * 12; c = _num(contributions_monthly)
    fv = p * (1 + r) ** n + (c * (((1 + r) ** n - 1) / r) if r else c * n)
    return {"future_value": round(fv, 2), "contributed": round(p + c * n, 2)}


@tool
def risk_allocation(age: int, risk_tolerance: str = "moderate") -> dict:
    """Suggested stock/bond split by age + risk tolerance.

    Args:
        age: investor age
        risk_tolerance: conservative | moderate | aggressive
    """
    base = max(20, min(90, 110 - int(age)))
    adj = {"conservative": -15, "moderate": 0, "aggressive": 15}.get(risk_tolerance.lower(), 0)
    eq = max(10, min(95, base + adj))
    return {"stocks_pct": eq, "bonds_pct": 100 - eq, "tolerance": risk_tolerance}


# =========================== RESEARCH (real) =================================

@tool
async def wikipedia_search(query: str) -> list:
    """Search Wikipedia article titles (real).

    Args:
        query: search terms
    """
    try:
        data = await _get_json("https://en.wikipedia.org/w/api.php",
                               {"action": "query", "list": "search", "srsearch": query,
                                "format": "json", "srlimit": 5})
        return [h["title"] for h in data["query"]["search"]]
    except Exception as e:  # noqa: BLE001
        return [f"search error: {e}"]


@tool
async def wikipedia_summary(title: str) -> dict:
    """Fetch the lead summary of a Wikipedia article (real).

    Args:
        title: exact article title
    """
    try:
        import urllib.parse
        t = urllib.parse.quote(title.replace(" ", "_"))
        data = await _get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{t}")
        return {"title": data.get("title"), "extract": (data.get("extract") or "")[:900],
                "url": data.get("content_urls", {}).get("desktop", {}).get("page")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"summary error: {e}"}


@tool
async def arxiv_search(query: str) -> list:
    """Search arXiv for recent papers (real; returns titles + summaries).

    Args:
        query: research topic
    """
    try:
        async with httpx.AsyncClient(timeout=15, headers=UA) as c:
            r = await c.get("http://export.arxiv.org/api/query",
                            params={"search_query": f"all:{query}", "max_results": 4,
                                    "sortBy": "submittedDate", "sortOrder": "descending"})
            r.raise_for_status()
        import re
        titles = re.findall(r"<title>(.*?)</title>", r.text, re.S)[1:]  # skip feed title
        return [t.strip().replace("\n", " ")[:160] for t in titles]
    except Exception as e:  # noqa: BLE001
        return [f"arxiv error: {e}"]
