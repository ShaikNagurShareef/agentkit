"""Computer / desktop use (§6.3). Disabled unless explicitly enabled.

ComputerDriver is the low-level surface (screenshot/click/type/key). E2BDesktopDriver
(remote managed desktop, preferred) and PyAutoGuiDriver (host control, UNSANDBOXED,
opt-in only) implement it. Multi-tenant or untrusted use must select E2B.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from ..context import RunContext
from ..errors import ToolError
from .base import Tool, ToolResult


@runtime_checkable
class ComputerDriver(Protocol):
    async def screenshot(self) -> bytes: ...
    async def click(self, x: int, y: int, button: str = "left") -> None: ...
    async def type(self, text: str) -> None: ...
    async def key(self, combo: str) -> None: ...


class PyAutoGuiDriver:
    """Drives the host machine. UNSANDBOXED, single-user, opt-in only."""

    def __init__(self) -> None:
        try:
            import pyautogui  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional dep
            raise ToolError(
                "computer use needs pyautogui: pip install 'agentkit[computer]'",
                where="computer",
            ) from e

    async def screenshot(self) -> bytes:  # pragma: no cover - host GUI
        import io

        import pyautogui

        buf = io.BytesIO()
        pyautogui.screenshot().save(buf, format="PNG")
        return buf.getvalue()

    async def click(self, x: int, y: int, button: str = "left") -> None:  # pragma: no cover
        import pyautogui

        pyautogui.click(x, y, button=button)

    async def type(self, text: str) -> None:  # pragma: no cover
        import pyautogui

        pyautogui.typewrite(text)

    async def key(self, combo: str) -> None:  # pragma: no cover
        import pyautogui

        pyautogui.hotkey(*combo.split("+"))


class E2BDesktopDriver:
    """Remote managed desktop (preferred, no local VM). Optional dependency."""

    def __init__(self) -> None:  # pragma: no cover - requires E2B desktop
        try:
            import e2b_desktop  # noqa: F401
        except ImportError as e:
            raise ToolError(
                "remote desktop needs e2b-desktop: pip install 'agentkit[sandbox]'",
                where="computer",
            ) from e


def build_driver(kind: Literal["e2b", "pyautogui"] = "e2b") -> ComputerDriver:
    return E2BDesktopDriver() if kind == "e2b" else PyAutoGuiDriver()


def computer_tool(*, enable_computer_use: bool = False, driver: Literal["e2b", "pyautogui"] = "e2b") -> Tool:
    """A `computer` Tool. Raises unless ``enable_computer_use=True`` (§6.3)."""
    if not enable_computer_use:
        raise ToolError(
            "computer use is disabled; pass enable_computer_use=True to enable", where="computer"
        )
    drv = build_driver(driver)

    async def handler(args: dict, ctx: RunContext) -> ToolResult:
        action = args.get("action")
        if action == "screenshot":
            data = await drv.screenshot()
            return ToolResult(ok=True, content=f"<screenshot {len(data)} bytes>")
        if action == "click":
            await drv.click(int(args["x"]), int(args["y"]), args.get("button", "left"))
        elif action == "type":
            await drv.type(args.get("text", ""))
        elif action == "key":
            await drv.key(args.get("combo", ""))
        else:
            return ToolResult(ok=False, content=f"unknown action '{action}'")
        return ToolResult(ok=True, content="ok")

    return Tool(
        name="computer",
        description="Control a desktop: screenshot / click / type / key.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["screenshot", "click", "type", "key"]},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "text": {"type": "string"},
                "combo": {"type": "string"},
            },
            "required": ["action"],
        },
        source="local",
        timeout_s=120,
        handler=handler,
    )
