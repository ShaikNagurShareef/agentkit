"""Tool registry + entry-point discovery (§6.4).

Third parties ship tools as packages exposing the ``agentkit.tools`` entry point;
they appear by name in any agent spec.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from .base import Tool

ENTRYPOINT_GROUP = "agentkit.tools"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"tool '{name}' is not registered")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def discover_entrypoints(self) -> None:
        """Load tools advertised under the ``agentkit.tools`` entry point.

        Each entry point may resolve to a ``Tool`` or to a callable that, when
        called with no args, returns a ``Tool``.
        """
        for ep in entry_points(group=ENTRYPOINT_GROUP):
            obj = ep.load()
            tool = obj() if callable(obj) and not isinstance(obj, Tool) else obj
            if isinstance(tool, Tool):
                self.register(tool)


# Process-wide default registry.
DEFAULT_REGISTRY = ToolRegistry()
