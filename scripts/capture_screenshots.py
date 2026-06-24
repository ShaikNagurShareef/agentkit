"""Capture dashboard + Agent UI screenshots for the README (dev utility).

Boots example apps, drives them with Playwright (headless Chromium), and saves
PNGs to assets/screenshots/. Requires GEMINI_API_KEY / OPENAI_API_KEY in .env.

    python -m playwright install chromium
    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable


def wait(port: int, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.4)
    raise RuntimeError(f"port {port} not ready")


def boot(app: str, port: int) -> subprocess.Popen:
    env = {**os.environ, "PORT": str(port), "DASHBOARD_PORT": str(port + 100)}
    p = subprocess.Popen([PY, f"examples/{app}"], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait(port); wait(port + 100)
    return p


def shot(page, path: Path, full=True):
    page.screenshot(path=str(path), full_page=full)
    print("  saved", path.relative_to(ROOT))


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)

        # ---------------- Healthcare: Agent UI + dashboard ----------------
        proc = boot("app_healthcare.py", 8811)
        try:
            # Agent UI (chat)
            pg = ctx.new_page()
            pg.goto("http://127.0.0.1:8811/")
            pg.wait_for_selector("#examples .ex", timeout=15000)
            pg.fill("#input", "I'm 68 with sudden chest pain and shortness of breath; I take warfarin and was just prescribed aspirin. How urgent is this?")
            pg.click("#send")
            pg.wait_for_function("document.querySelector('.msg.assistant .bubble')?.textContent.length>40", timeout=90000)
            pg.wait_for_timeout(1500)
            shot(pg, OUT / "agent-ui-healthcare.png")
            pg.close()

            # Dashboard — run, then capture Run / Components / Memory
            db = ctx.new_page()
            db.goto("http://127.0.0.1:8811".replace("8811", "8911") + "/")
            db.wait_for_selector("#run", timeout=15000)
            db.fill("#input", "What are the FDA warnings for metformin, and the ICD-10 code for type 2 diabetes?")
            db.click("#run")
            db.wait_for_function("document.querySelectorAll('#graph .gnode').length>2", timeout=90000)
            db.wait_for_function("document.querySelector('#out')?.textContent.length>30", timeout=90000)
            db.wait_for_timeout(1200)
            shot(db, OUT / "dashboard-run-trace.png")
            db.click('.tab[data-v="components"]'); db.wait_for_timeout(700)
            shot(db, OUT / "dashboard-components.png")
            db.click('.tab[data-v="memory"]'); db.wait_for_timeout(1200)
            shot(db, OUT / "dashboard-memory.png")
            db.close()
        finally:
            proc.terminate(); proc.wait(timeout=10)

        # ---------------- Finance: multi-agent flow trace ----------------
        proc = boot("app_finance.py", 8812)
        try:
            db = ctx.new_page()
            db.goto("http://127.0.0.1:8912/")
            db.wait_for_selector("#run", timeout=15000)
            db.fill("#input", "I'm 35, earn $9,000/month, $1,200/month debt, $50k saved, can invest $1,000/month. Buy a $400k home in 2 years; retire by 60. Moderate risk.")
            db.click("#run")
            db.wait_for_function("document.querySelectorAll('#graph .gnode').length>=4", timeout=120000)
            db.wait_for_function("document.querySelector('#out')?.textContent.length>30", timeout=120000)
            db.wait_for_timeout(1500)
            shot(db, OUT / "dashboard-finance-flow.png")
            db.click('.tab[data-v="graph"]'); db.wait_for_timeout(900)
            shot(db, OUT / "dashboard-finance-topology.png")
            db.close()
        finally:
            proc.terminate(); proc.wait(timeout=10)

        # ---------------- Supervisor: hierarchical multi-agent ----------------
        proc = boot("app_supervisor.py", 8815)
        try:
            db = ctx.new_page()
            db.goto("http://127.0.0.1:8915/")
            db.wait_for_selector("#run", timeout=15000)
            db.fill("#input", "Find the populations of Japan and Germany, compute the combined total, and write a one-sentence summary.")
            db.click("#run")
            db.wait_for_function("document.querySelector('#out')?.textContent.length>30", timeout=150000)
            db.wait_for_timeout(1500)
            shot(db, OUT / "dashboard-supervisor.png")
            db.close()
        finally:
            proc.terminate(); proc.wait(timeout=10)

        browser.close()
    print("done.")


if __name__ == "__main__":
    main()
