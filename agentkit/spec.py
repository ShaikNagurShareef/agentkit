"""Declarative spec loading (§12.2, §12.3): build Agents/Flows from YAML.

`apiVersion: agentkit/v1` documents of `kind: Agent` or `kind: Flow`. Tool refs
resolve to built-in capability tools or registered/entry-point tools.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .flow import FlowSpec, NodeSpec
    from .tools.base import Tool


def _read_doc(path: str | Path) -> dict:
    text = Path(path).read_text()
    if str(path).endswith((".json",)):
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("YAML specs need pyyaml: pip install 'agentkit[yaml]'") from e
    return yaml.safe_load(text)


def resolve_tool(entry: Any) -> "Tool":
    """Resolve a tool spec entry to a Tool (built-ins, registry, or entry points)."""
    from .tools.base import Tool
    from .tools.registry import DEFAULT_REGISTRY

    ref = entry if isinstance(entry, str) else entry.get("ref")
    config = {} if isinstance(entry, str) else entry.get("config", {})

    if ref == "browser":
        from .tools.browser import BrowserTool

        return BrowserTool(**config).as_tool()
    if ref in ("code", "run_code"):
        from .tools.code import code_tool

        return code_tool(**config)
    if DEFAULT_REGISTRY.has(ref):
        return DEFAULT_REGISTRY.get(ref)
    # try import path "module:attr"
    if ":" in ref:
        module_path, _, attr = ref.partition(":")
        obj = getattr(importlib.import_module(module_path), attr)
        return obj if isinstance(obj, Tool) else obj
    # last resort: entry-point discovery
    DEFAULT_REGISTRY.discover_entrypoints()
    if DEFAULT_REGISTRY.has(ref):
        return DEFAULT_REGISTRY.get(ref)
    raise ValueError(f"could not resolve tool '{ref}'")


def load_agent_spec(path: str | Path):
    """Build an Agent from a `kind: Agent` YAML spec."""
    from .agent import Agent

    doc = _read_doc(path)
    spec = doc.get("spec", doc)
    name = doc.get("metadata", {}).get("name") or spec.get("name", "agent")
    tools = [resolve_tool(t) for t in spec.get("tools", [])]
    mcp_servers = [m["url"] if isinstance(m, dict) else m for m in spec.get("mcp_servers", [])]
    return Agent(
        name=name,
        model=spec["model"],
        instructions=spec.get("instructions"),
        tools=tools,
        mcp_servers=mcp_servers,
        a2a_peers=list(spec.get("a2a_peers", [])),
        memory=spec.get("memory"),
        observability=spec.get("observability", "langfuse"),
        max_steps=spec.get("max_steps", 20),
        step_timeout_s=spec.get("step_timeout_s", 120),
        on_tool_error=spec.get("on_tool_error", "surface"),
    )


def load_flow_spec(path: str | Path) -> "FlowSpec":
    """Parse a `kind: Flow` YAML spec into a FlowSpec."""
    from .flow import FlowSpec

    doc = _read_doc(path)
    spec = doc.get("spec", doc)
    name = doc.get("metadata", {}).get("name") or spec.get("name", "flow")
    return FlowSpec(name=name, **{k: spec[k] for k in ("nodes", "edges", "entry", "finish")})


def resolve_ref(node: "NodeSpec"):
    """Resolve a flow NodeSpec to a runnable (Agent / Flow / callable)."""
    ref = node.ref
    if node.kind == "agent":
        if ref.endswith((".yaml", ".yml", ".json")):
            return load_agent_spec(ref)
        module_path, _, attr = ref.partition(":")
        return getattr(importlib.import_module(module_path), attr)
    if node.kind == "flow":
        if ref.endswith((".yaml", ".yml", ".json")):
            from .flow import Flow

            return Flow.from_spec(load_flow_spec(ref))
        module_path, _, attr = ref.partition(":")
        return getattr(importlib.import_module(module_path), attr)
    # function
    module_path, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module_path), attr)


def load_target(target: str):
    """Load an Agent or Flow from a YAML/JSON spec file or a `module:attr` ref."""
    if target.endswith((".yaml", ".yml", ".json")):
        doc = _read_doc(target)
        kind = doc.get("kind", "Agent").lower()
        if kind == "flow":
            from .flow import Flow

            return Flow.from_spec(load_flow_spec(target))
        return load_agent_spec(target)
    module_path, _, attr = target.partition(":")
    if not attr:
        raise SystemExit("target must be 'module:attribute' or a .yaml spec file")
    return getattr(importlib.import_module(module_path), attr)
