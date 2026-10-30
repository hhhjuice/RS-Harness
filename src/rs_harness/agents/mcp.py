"""Lazy MCP tools in the existing LangChain agent loop."""

import json

from typing import Annotated, Any

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
    mcp_connections: Annotated[dict[str, dict[str, Any]], merge_connections]


class MCPMiddleware(AgentMiddleware):
    state_schema = MCPState

    def __init__(self, registry: MCPRegistry):
        self.registry = registry
        self.catalog_prompt = "\n\n## Available MCP servers\n" + "\n".join(
            f"- {name}" for name in registry.config.servers
        ) + (
            "\nCall connect_mcp with an exact server ID to discover its actual tools, resources and prompts. "
            "Wait for the connection result before calling server capabilities in the next model turn. "
            "Use only capabilities shown in the connected catalog; a configured server need not support every capability. "
            "An enabled server need not keep a live connection. Tools, resources and prompts may require host approval. "
            "Interactive MCP input (elicitation) is unsupported and is cancelled."
        )

        @tool
        async def connect_mcp(name: str, runtime: ToolRuntime[AgentContext]) -> Command:
            """Discover and enable tools from a host-configured MCP server by its exact ID."""
            catalog = await registry.connect(name, runtime.state)
            already = name in runtime.state.get("mcp_connections", {})
            text = f"MCP server {name}: {'already connected' if already else 'connected'}; {len(catalog.entries)} tools."
            summary = {
                "tools": sorted(catalog.entries),
                "resources": catalog.record["resources"],
                "prompts": catalog.record["prompts"],
            }
            full = text + "\nAvailable next turn: " + json.dumps(
                summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            content, _ = bounded_output(full, None, registry.max_output_bytes)
            if content != full:
                fallback = (
                    f"catalog_incomplete: tools={len(catalog.entries)}; "
                    f"resources={len(summary['resources']['items'])} ({summary['resources']['status']}); "
                    f"prompts={len(summary['prompts']['items'])} ({summary['prompts']['status']}). "
                    "See the connected catalog next turn."
                )
                content = bounded_error(fallback, registry.max_output_bytes)
            return Command(update={
                "mcp_connections": {name: catalog.record},
                "messages": [ToolMessage(content=content, name="connect_mcp", tool_call_id=runtime.tool_call_id)],
            })

        @tool
        async def read_mcp_resource(
            server: str, uri: str, runtime: ToolRuntime[AgentContext], offset: int = 0, limit: int | None = None,
        ) -> str:
            """Read a resource page with provenance; use next_offset to continue and limit to bound total UTF-8 bytes."""
            return await registry.read_resource(server, uri, runtime.state, offset=offset, limit=limit)

        @tool
        async def get_mcp_prompt(
            server: str, name: str, runtime: ToolRuntime[AgentContext],
            arguments: dict[str, Any] | None = None, offset: int = 0, limit: int | None = None,
        ) -> str:
            """Retrieve a rendered prompt page with message roles; use next_offset to continue."""
            return await registry.get_prompt(server, name, arguments, runtime.state, offset=offset, limit=limit)

        for item in (connect_mcp, read_mcp_resource, get_mcp_prompt):
            item.handle_tool_error = lambda error: bounded_error(error, registry.max_output_bytes)
        self.tools = [connect_mcp, read_mcp_resource, get_mcp_prompt] if registry.config.servers else []

    def wrap_model_call(self, request, handler):
        self.registry.check_sync(request.state)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        await self.registry.ensure_loaded(request.state)
        tools = [entry.tool for entry in self.registry.entries(request.state).values()]
        return await handler(self.with_catalog(request, tools))

    def _catalog_for_state(self, state) -> str:
        lines = []
        for name, record in sorted(state.get("mcp_connections", {}).items()):
            lines.append(f"- {name}")
            for kind, tool_name in (("resources", "read_mcp_resource"), ("prompts", "get_mcp_prompt")):
                capability = record.get(kind, {"status": "error", "items": []})
                status = capability.get("status", "error")
                items = capability.get("items", [])
                if status == "ok" and items:
                    definitions = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    lines.append(f"  - {kind} (use {tool_name}): {definitions}")
                elif status == "ok":
                    lines.append(f"  - {kind}: empty")
                else:
                    lines.append(f"  - {kind} discovery: {status}")
        if not lines:
            return self.catalog_prompt
        return self.catalog_prompt + "\n\n## Connected MCP capabilities\n" + "\n".join(lines)

    def with_catalog(self, request, tools):
        """Pure request projection shared with offline context inspection."""
        if not self.registry.config.servers:
            return request
        system = request.system_message or SystemMessage(content="")
        content = [*system.content_blocks, {"type": "text", "text": self._catalog_for_state(request.state)}]
        return request.override(tools=[*request.tools, *tools], system_message=system.model_copy(update={"content": content}))

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
