"""Render the layered architecture as a static stacked figure (PNG) for the README.

Each layer is a horizontal band holding its components as chips, stacked top→bottom
with arrows between — the original stacked look, but as an image so it renders
identically everywhere (GitHub, PyPI) without a Mermaid engine.

    python scripts/make_architecture_figure.py   # -> assets/architecture.png
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# (layer title, subtitle, [component chips]) — top of the stack first.
LAYERS = [
    ("Public SDK API", "what you import", ["Agent", "Flow", "@tool / Tool", "EvalRunner", "serve()"]),
    ("Engine", "compile & run", ["GraphCompiler", "CompiledGraph · LangGraph", "AgentState channels", "Executor"]),
    ("Runtime", "serve & persist", ["create_app · FastAPI", "Agent UI + Dashboard", "Checkpointer", "SessionManager"]),
    ("Protocols", "interop", ["MCP client", "MCP server", "A2A client / server", "Agent Card"]),
    ("Capabilities", "what agents can do", ["MemoryStore", "Tools: code / browser / computer", "Tracer + Eval", "Identity / Secrets"]),
    ("Providers", "official SDKs", ["Anthropic", "OpenAI", "Gemini", "Groq"]),
]

bands = "\n".join(
    f"""
    <div class="layer">
      <div class="label"><div class="title">{title}</div><div class="sub">{sub}</div></div>
      <div class="chips">{''.join(f'<span class="chip">{c}</span>' for c in chips)}</div>
    </div>
    {'<div class="arrow">&#9660;</div>' if i < len(LAYERS) - 1 else ''}"""
    for i, (title, sub, chips) in enumerate(LAYERS)
)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #ffffff; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  #fig {{ width: 940px; padding: 28px 30px 30px; background:
          radial-gradient(140% 100% at 50% 0%, #fbfcff 0%, #f4f6fb 100%); }}
  .heading {{ font-size: 13px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
              color: #64748b; margin: 0 4px 16px; }}
  .layer {{ display: flex; align-items: stretch; border: 1px solid #dfe3ee;
            border-radius: 12px; overflow: hidden; box-shadow: 0 1px 2px rgba(15,23,42,.05);
            background: #ffffff; }}
  .label {{ flex: 0 0 196px; padding: 14px 18px; display: flex; flex-direction: column;
            justify-content: center; gap: 3px;
            background: linear-gradient(180deg, #eef2ff 0%, #e7ecff 100%);
            border-right: 1px solid #d7dcef; }}
  .title {{ font-size: 16px; font-weight: 700; color: #312e81; }}
  .sub {{ font-size: 11.5px; color: #6b7280; }}
  .chips {{ flex: 1; display: flex; flex-wrap: wrap; align-content: center; gap: 8px;
            padding: 14px 18px; }}
  .chip {{ font-size: 12.5px; color: #1f2937; background: #f3f5fb; border: 1px solid #d9deea;
           border-radius: 999px; padding: 5px 12px; white-space: nowrap; }}
  .arrow {{ text-align: center; color: #aab2c5; font-size: 13px; line-height: 1; margin: 7px 0; }}
</style></head><body>
  <div id="fig">
    <div class="heading">AgentKit &mdash; layered architecture</div>
    {bands}
  </div>
</body></html>"""

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_context(device_scale_factor=2).new_page()
    page.set_content(HTML, wait_until="networkidle")
    page.locator("#fig").screenshot(path=str(OUT))
    browser.close()

print("saved", OUT.relative_to(ROOT))
