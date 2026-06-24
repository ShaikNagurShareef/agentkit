"""M7: a multi-step Flow composing functions and agents (UC-3).

This pipeline runs with no API key (pure-function nodes) to demonstrate control
flow. Swap any node for an `Agent` to make it model-driven, then `flow.serve()`.

    agentkit run examples.flow_pipeline:flow --input "  Hello   World  "
    agentkit serve examples.flow_pipeline:flow
"""

from __future__ import annotations

from agentkit import Flow


def normalize(text: str) -> str:
    return " ".join(text.split())


def headline(text: str) -> str:
    return text.title()


# researcher -> writer style pipeline; here both steps are plain functions.
flow = (
    Flow("normalize-and-headline")
    .step(normalize, name="normalize")
    .step(headline, name="headline")
    .when(lambda v: len(v) > 40)
    .then(lambda v: v[:40] + "…", name="truncate")
    .otherwise(lambda v: v, name="passthrough")
)


if __name__ == "__main__":
    print(flow.run("  the   quick brown   fox  ").output)
