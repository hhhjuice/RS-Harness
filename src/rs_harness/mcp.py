"""Host-owned MCP discovery, routing and recoverable catalog records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import anyio
import httpx2
from fastmcp.client import Client
from fastmcp.client.auth.bearer import BearerAuth
from fastmcp.client.elicitation import ElicitResult
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from fastmcp.exceptions import ClientError
from langchain_core._api import LangChainBetaWarning
from langchain_core.tools import BaseTool, ToolException
from mcp import MCPError
from mcp.client import InputRequiredRoundsExceededError
from pydantic import ValidationError

from .config import MCPConfig, MCPStdioConfig, mcp_name


class MCPRecoveryError(RuntimeError):
    """The saved catalog cannot safely be used for this interaction."""


@dataclass(frozen=True)
class MCPEntry:
    server_id: str
    raw_name: str
    tool: BaseTool


@dataclass(frozen=True)
class MCPCatalog:
    record: dict[str, str]
    entries: dict[str, MCPEntry]


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exposed_name(server_id: str, raw_name: str) -> str:
    name = f"mcp__{mcp_name(server_id)}__{mcp_name(raw_name)}"
    if not server_id or not raw_name or len(name) > 64:
        raise ToolException("mcp_invalid_name: tool names must be non-empty and aliases at most 64 characters")
    return name


def _external_failure(error: BaseException) -> bool:
    if isinstance(error, (OSError, httpx2.HTTPError, MCPError, ClientError, ValidationError,
                          anyio.EndOfStream, anyio.BrokenResourceError, anyio.ClosedResourceError)):
        return True
    if isinstance(error, BaseExceptionGroup):
        return bool(error.exceptions) and all(_external_failure(item) for item in error.exceptions)
    return isinstance(error, RuntimeError) and error.__cause__ is not None and _external_failure(error.__cause__)


def bounded_output(content, artifact, limit: int):
    """Bound the payload before tool callbacks and checkpoint serialization."""
    size = len(content.encode()) if isinstance(content, str) else len(json.dumps(content, ensure_ascii=False).encode())
    if artifact is not None:
        size += len(json.dumps(artifact, ensure_ascii=False).encode())
    if size <= limit:
        return content, artifact
    notice = "MCP result received; output omitted because it exceeds max_tool_output_bytes."
    return notice.encode()[:limit].decode("utf-8", errors="ignore"), None


def bounded_error(error: Exception | str, limit: int) -> str:
    """Keep the error's meaning; truncation must not claim a result was received."""
    return str(error).encode("utf-8")[:limit].decode("utf-8", errors="ignore")


async def cancel_elicitation(message, response_type, params, context):
    """RSH has no interactive MCP input flow; never invoke LangGraph interrupt."""
    return ElicitResult(action="cancel", _meta={"reason": "RSH does not support interactive MCP input"})


class MCPRegistry:
    def __init__(self, config: MCPConfig, *, local_names=(), max_output_bytes: int = 65536):
        self.config = config
        self.local_names = set(local_names)
        self.max_output_bytes = max_output_bytes
        self.catalogs: dict[str, MCPCatalog] = {}
        self._locks = {name: asyncio.Lock() for name in config.servers}

    def _config_digest(self, name: str) -> str:
        # Environment references, not their current secret values, define identity.
        return _digest({"server": self.config.servers[name].model_dump(mode="json"), "timeout_s": self.config.timeout_s})

    def _make_adapter(self, name: str):
        # Import only on connection. The pinned beta API is documented; an import
        # advisory must not break the CLI's quiet/JSON output contract.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LangChainBetaWarning, message="`langchain.mcp` is in beta.*")
            from langchain.mcp import MCPAdapter

        definition = self.config.servers[name]

        def environment(reference: str) -> str:
            value = os.environ.get(reference)
            if value is None:
                raise ToolException(f"mcp_missing_environment: {reference}")
            return value

        if isinstance(definition, MCPStdioConfig):
            transport = StdioTransport(
                command=definition.command, args=definition.args, cwd=str(definition.cwd),
                env={key: environment(ref) for key, ref in definition.env_from_env.items()} or None,
                keep_alive=False, log_file=Path(os.devnull),
            )
        else:
            token = environment(definition.bearer_token_env) if definition.bearer_token_env else None
            transport = StreamableHttpTransport(str(definition.url), auth=BearerAuth(token) if token is not None else None)
        return MCPAdapter(Client(transport, timeout=self.config.timeout_s,
                                 init_timeout=self.config.timeout_s, elicitation_handler=cancel_elicitation))

    def _tool_copy(self, original: BaseTool, name: str) -> BaseTool:
        # Keep the SDK's schema, raw-name closure, result conversion and error handler.
        # This boundary runs *inside* BaseTool so callbacks never see raw transport errors.
        async def invoke(**arguments):
            try:
                async with asyncio.timeout(self.config.timeout_s):
                    content, artifact = await original.coroutine(**arguments)
            except InputRequiredRoundsExceededError:
                raise ToolException("mcp_input_required_limit: outcome may be unknown; server exceeded the SDK continuation limit") from None
            except Exception as exc:
                if not _external_failure(exc):
                    raise
                raise ToolException(
                    "mcp_call_failed: outcome may be unknown; transport failed or timed out. "
                    "Check its state before retrying. No automatic retry was performed."
                ) from None
            return bounded_output(content, artifact, self.max_output_bytes)

        def handle_error(error):
            try:
                content = original.handle_tool_error(error)
            except ToolException as exc:
                return bounded_error(exc, self.max_output_bytes)
            return bounded_output(content, None, self.max_output_bytes)[0]

        return original.model_copy(update={"name": name, "coroutine": invoke, "handle_tool_error": handle_error})

    async def _load(self, name: str, expected: dict[str, str] | None = None) -> MCPCatalog:
        async with self._locks[name]:
            cached = self.catalogs.get(name)
            if cached is not None:
                if expected is not None and cached.record != expected:
                    raise MCPRecoveryError(f"mcp_catalog_mismatch: {name}; create a new session")
                return cached
            try:
                async with asyncio.timeout(self.config.timeout_s):
                    tools = await self._make_adapter(name).list_tools(cache_mode="bypass")
            except Exception as exc:
                if isinstance(exc, ToolException):
                    raise
                if not _external_failure(exc):
                    raise
                raise ToolException(f"mcp_connect_failed: {name}; discovery failed or timed out") from None
            entries = {}
            definitions = []
            occupied = self.local_names | {alias for catalog in self.catalogs.values() for alias in catalog.entries}
            for original in tools:
                alias = exposed_name(name, original.name)
                if alias in occupied or alias in entries:
                    raise ToolException(f"mcp_name_conflict: {alias}; no tools from this connection were registered")
                schema = original.args_schema
                if not isinstance(schema, dict):
                    schema = schema.model_json_schema()
                definitions.append({"raw_name": original.name, "name": alias,
                                    "description": original.description, "args_schema": schema})
                entries[alias] = MCPEntry(name, original.name, self._tool_copy(original, alias))
            record = {"config_digest": self._config_digest(name),
                      "catalog_digest": _digest(sorted(definitions, key=lambda item: item["name"]))}
            if expected is not None and record != expected:
                raise MCPRecoveryError(f"mcp_catalog_mismatch: {name}; create a new session")
            catalog = MCPCatalog(record, entries)
            # No await between the global collision check and this atomic commit.
            self.catalogs[name] = catalog
            return catalog

    async def ensure_loaded(self, state) -> None:
        records = state.get("mcp_connections", {})
        # Validate *all* configuration before connecting to any of the saved servers.
        for name, record in records.items():
            if name not in self.config.servers or self._config_digest(name) != record["config_digest"]:
                raise MCPRecoveryError(f"mcp_config_mismatch: {name}; restore the configuration or create a new session")
        for name, record in records.items():
            try:
                await self._load(name, record)
            except ToolException as exc:
                raise MCPRecoveryError(f"mcp_restore_failed: {name}; {exc}. Retry after fixing the service") from None

    async def connect(self, name: str, state) -> MCPCatalog:
        await self.ensure_loaded(state)
        if name not in self.config.servers:
            raise ToolException("mcp_unknown_server: choose one of " + ", ".join(self.config.servers))
        return await self._load(name)

    def entries(self, state) -> dict[str, MCPEntry]:
        # An in-flight connect can warm the cache before its Command reaches state.
        return {alias: entry for name in state.get("mcp_connections", {})
                for alias, entry in self.catalogs[name].entries.items()}

    def requires_approval(self, entry: MCPEntry) -> bool:
        return self.config.servers[entry.server_id].tools.get(entry.raw_name, "allow") == "confirm"

    def check_sync(self, state) -> None:
        if state.get("mcp_connections"):
            for name, record in state["mcp_connections"].items():
                if name not in self.config.servers or self._config_digest(name) != record["config_digest"]:
                    raise MCPRecoveryError(f"mcp_config_mismatch: {name}; create a new session")
        if self.config.servers or state.get("mcp_connections"):
            raise MCPRecoveryError("mcp_async_required: use ainvoke for MCP-enabled sessions")
