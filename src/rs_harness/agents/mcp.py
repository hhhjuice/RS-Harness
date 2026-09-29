"""Lazy MCP tools in the existing LangChain agent loop."""

from typing import Annotated

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.types import Command

from ..context import AgentContext
from ..mcp import MCPRecoveryError, MCPRegistry, bounded_error, bounded_output


def merge_connections(current: dict, update: dict) -> dict:
    merged = dict(current)
    for name, record in update.items():
        if name in merged and merged[name] != record:
            raise MCPRecoveryError(f"mcp_connection_conflict: {name}")
        merged[name] = record
    return merged


class MCPState(AgentState):
    mcp_connections: Annotated[dict[str, dict[str, str]], merge_connections]


class MCPMiddleware(AgentMiddleware):
    state_schema = MCPState

    def __init__(self, registry: MCPRegistry):
        self.registry = registry
        self.catalog_prompt = "\n\n## Available MCP servers\n" + "\n".join(
            f"- {name}" for name in registry.config.servers
        ) + (
            "\nCall connect_mcp with an exact server ID to discover its tools when needed. "
            "Wait for the result before calling those tools in the next model turn. "
            "An enabled server need not keep a live connection. Tools may require host approval. "
            "Interactive MCP input (elicitation) is unsupported and is cancelled."
        )

        @tool
        async def connect_mcp(name: str, runtime: ToolRuntime[AgentContext]) -> Command:
            """Discover and enable tools from a host-configured MCP server by its exact ID."""
            catalog = await registry.connect(name, runtime.state)
            already = name in runtime.state.get("mcp_connections", {})
            text = f"MCP server {name}: {'already connected' if already else 'connected'}; {len(catalog.entries)} tools."
            names = ", ".join(catalog.entries)
            content, _ = bounded_output(text + "\nAvailable next turn: " + names, None, registry.max_output_bytes)
            if content != text + "\nAvailable next turn: " + names:
                content = text.encode()[:registry.max_output_bytes].decode("utf-8", errors="ignore")
            return Command(update={
                "mcp_connections": {name: catalog.record},
                "messages": [ToolMessage(content=content, name="connect_mcp", tool_call_id=runtime.tool_call_id)],
            })

        connect_mcp.handle_tool_error = lambda error: bounded_error(error, registry.max_output_bytes)
        self.tools = [connect_mcp] if registry.config.servers else []

    def wrap_model_call(self, request, handler):
        self.registry.check_sync(request.state)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        await self.registry.ensure_loaded(request.state)
        if not self.registry.config.servers:
            return await handler(request)
        system = request.system_message or SystemMessage(content="")
        content = [*system.content_blocks, {"type": "text", "text": self.catalog_prompt}]
        tools = [*request.tools, *(entry.tool for entry in self.registry.entries(request.state).values())]
        return await handler(request.override(tools=tools, system_message=system.model_copy(update={"content": content})))

    async def awrap_tool_call(self, request, handler):
        await self.registry.ensure_loaded(request.state)
        if not request.tool_call["name"].startswith("mcp__"):
            return await handler(request)
        entry = self.registry.entries(request.state).get(request.tool_call["name"])
        if entry is None:
            raise ToolException(bounded_error(
                "mcp_unknown_tool: connect the configured server first, then use an advertised alias in the next turn",
                self.registry.max_output_bytes,
            ))
        return await handler(request.override(tool=entry.tool))

    def wrap_tool_call(self, request, handler):
        self.registry.check_sync(request.state)
        return handler(request)
