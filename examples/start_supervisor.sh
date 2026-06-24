#!/usr/bin/env bash
# Hierarchical supervisor — delegates to researcher / analyst / writer subagents.
set -e
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -q -e '.[openai,gemini,yaml,langfuse]'
export PORT="${PORT:-8815}"
URL="http://127.0.0.1:${PORT}"
echo "▶ Supervisor — Agent UI: ${URL}   ·   Orchestrator dashboard: http://127.0.0.1:$((PORT+100))"
( sleep 2.5; .venv/bin/python -c "import webbrowser;webbrowser.open('${URL}')" >/dev/null 2>&1 & )
exec .venv/bin/python examples/app_supervisor.py
