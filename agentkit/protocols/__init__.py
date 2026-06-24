"""Interoperability protocols: MCP (consume + expose) and A2A (peer + expose).

MCP turns external services into tools and the agent's own tools into a server;
A2A lets the agent act as, and call, peer agents.
"""

from .a2a import A2AClient, A2AServer, AgentCard
from .mcp_client import MCPClient, MCPToolDescriptor
from .mcp_server import MCPServer

__all__ = [
    "MCPClient",
    "MCPToolDescriptor",
    "MCPServer",
    "A2AServer",
    "A2AClient",
    "AgentCard",
]
