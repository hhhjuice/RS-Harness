"""Host-owned MCP discovery, routing and recoverable catalog records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio
import httpx2
from fastmcp.client import Client
from fastmcp.client.auth.bearer import BearerAuth
from fastmcp.client.elicitation import ElicitResult
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from fastmcp.exceptions import ClientError
from langchain_core._api import LangChainBetaWarning
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp import MCPError
from mcp.client import InputRequiredRoundsExceededError
from mcp.types import METHOD_NOT_FOUND
from pydantic import AnyUrl, ValidationError

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
    record: dict[str, Any]
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


def _text_chunks(text: str, size: int = 2048):
    for start in range(0, len(text), size):
        yield text[start:start + size]


def _resource_chunks(contents):
    if not contents:
        yield "MCP resource returned no content parts."
        return
    for index, item in enumerate(contents):
        metadata = {
            "index": index, "uri": str(item.uri), "mime_type": item.mime_type,
        }
        yield "[resource part " + json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "]\n"
        text = getattr(item, "text", None)
        if isinstance(text, str):
            if text:
                yield from _text_chunks(text)
            else:
                yield "[empty text]"
        else:
            yield f"[non-text content omitted: {type(item).__name__}]"
        yield "\n"


def _prompt_chunks(messages):
    if not messages:
        yield "MCP prompt returned no messages."
        return
    for index, message in enumerate(messages):
        content = message.content
        embedded = getattr(content, "resource", None)
        source = embedded if embedded is not None else content
        metadata = {
            "index": index, "role": str(message.role),
            "content_type": str(getattr(content, "type", type(content).__name__)),
        }
        uri = getattr(source, "uri", None)
        if uri is not None:
            metadata["uri"] = str(uri)
        mime_type = getattr(source, "mime_type", None)
        if mime_type is not None:
            metadata["mime_type"] = mime_type
        yield "[prompt message " + json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "]\n"
        text = getattr(source, "text", None)
        if isinstance(text, str):
            if text:
                yield from _text_chunks(text)
            else:
                yield "[empty text]"
        else:
            yield f"[non-text content omitted: {type(content).__name__}]"
        yield "\n"


def _utf8_prefix(content: bytes, size: int) -> tuple[str, int]:
    end = min(size, len(content))
    while end:
        try:
            return content[:end].decode("utf-8"), end
        except UnicodeDecodeError as exc:
            end = exc.start
    return "", 0


def _page_envelope(
    kind: str, content: bytes, size: int, *, offset: int, limit: int,
    total_bytes: int, item_count: int,
) -> tuple[str, int, int | None]:
    text, size = _utf8_prefix(content, size)
    next_offset = offset + size if offset + size < total_bytes else None
    envelope = {
        "kind": kind, "offset": offset, "limit": limit,
        "total_bytes": total_bytes, "item_count": item_count,
        "content": text, "truncated": next_offset is not None,
        "next_offset": next_offset,
        "continuation": (f"Repeat the same call with offset={next_offset}."
                         if next_offset is not None else None),
        "end": next_offset is None,
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")), size, next_offset


def _paged_mcp_output(
    kind: Literal["resource", "prompt"], chunks, item_count: int,
    offset: int, limit: int | None, max_output_bytes: int,
) -> str:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ToolException("mcp_invalid_offset: offset must be a non-negative integer")
    if limit is None:
        limit = max_output_bytes
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 or limit > max_output_bytes:
        raise ToolException(f"mcp_invalid_limit: limit must be between 1 and {max_output_bytes}")

    position = 0
    page_parts = []
    page_size = 0
    for chunk in chunks:
        encoded = chunk.encode("utf-8")
        end = position + len(encoded)
        if position <= offset < end:
            relative = offset - position
            try:
                encoded[:relative].decode("utf-8")
            except UnicodeDecodeError:
                raise ToolException("mcp_invalid_offset: offset must be on a UTF-8 character boundary") from None
        if end > offset and position < offset + limit and page_size < limit:
            start = max(offset - position, 0)
            fragment = encoded[start:start + limit - page_size]
            page_parts.append(fragment)
            page_size += len(fragment)
        position = end
    total_bytes = position
    if offset > total_bytes:
        raise ToolException(f"mcp_invalid_offset: offset {offset} exceeds total_bytes {total_bytes}")

    page = b"".join(page_parts)
    size = len(page)
    while True:
        rendered, actual_size, next_offset = _page_envelope(
            kind, page, size, offset=offset, limit=limit,
            total_bytes=total_bytes, item_count=item_count,
        )
        rendered_size = len(rendered.encode("utf-8"))
        if rendered_size <= limit:
            if next_offset == offset and offset < total_bytes:
                break
            return rendered
        if actual_size == 0:
            break
        ratio = min(1.0, limit / rendered_size)
        reduced = min(actual_size - 1, max(1, int(actual_size * ratio * 0.95)))
        _, size = _utf8_prefix(page, reduced)
    raise ToolException("mcp_output_limit_too_small: increase max_tool_output_bytes or limit")


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

    async def _run_sdk_operation(
        self, operation, *, failure: Literal["call", "connect"] = "call", name: str | None = None,
    ):
        try:
            async with asyncio.timeout(self.config.timeout_s):
                return await operation()
        except ToolException:
            raise
        except InputRequiredRoundsExceededError:
            if failure == "connect":
                raise ToolException(f"mcp_connect_failed: {name}; discovery failed or timed out") from None
            raise ToolException(
                "mcp_input_required_limit: outcome may be unknown; server exceeded the SDK continuation limit",
            ) from None
        except Exception as exc:
            if not _external_failure(exc):
                raise
            if failure == "connect":
                raise ToolException(f"mcp_connect_failed: {name}; discovery failed or timed out") from None
            raise ToolException(
                "mcp_call_failed: outcome may be unknown; transport failed or timed out. "
                "Check its state before retrying. No automatic retry was performed."
            ) from None

    def _tool_copy(self, original: BaseTool, name: str) -> BaseTool:
        # Keep the SDK's schema, raw-name closure, result conversion and error handler.
        # This boundary runs *inside* BaseTool so callbacks never see raw transport errors.
        async def invoke(**arguments):
            content, artifact = await self._run_sdk_operation(lambda: original.coroutine(**arguments))
            return bounded_output(content, artifact, self.max_output_bytes)

        def handle_error(error):
            try:
                content = original.handle_tool_error(error)
            except ToolException as exc:
                return bounded_error(exc, self.max_output_bytes)
            return bounded_output(content, None, self.max_output_bytes)[0]

        return original.model_copy(update={"name": name, "coroutine": invoke, "handle_tool_error": handle_error})

    async def _discover_optional(self, client: Client, kind: Literal["resources", "prompts"]) -> dict[str, Any]:
        try:
            if kind == "resources":
                discovered = await client.list_resources()
                definitions = sorted(({
                    "uri": str(item.uri), "name": item.name,
                    "description": item.description, "mime_type": item.mime_type,
                } for item in discovered), key=lambda item: item["uri"])
            else:
                discovered = await client.list_prompts()
                definitions = sorted(({
                    "name": item.name, "description": item.description,
                    "arguments": [{
                        "name": argument.name, "description": argument.description,
                        "required": argument.required,
                    } for argument in item.arguments or []],
                } for item in discovered), key=lambda item: item["name"])
        except MCPError as exc:
            status = "unsupported" if exc.code == METHOD_NOT_FOUND else "error"
            return {"status": status, "items": []}
        return {"status": "ok", "items": definitions}

    async def _load(self, name: str, expected: dict[str, Any] | None = None) -> MCPCatalog:
        async with self._locks[name]:
            cached = self.catalogs.get(name)
            if cached is not None:
                if expected is not None and cached.record != expected:
                    raise MCPRecoveryError(f"mcp_catalog_mismatch: {name}; create a new session")
                return cached
            async def discover():
                adapter = self._make_adapter(name)
                async with adapter:
                    tools = await adapter.list_tools(cache_mode="bypass")
                    resources = await self._discover_optional(adapter.client, "resources")
                    prompts = await self._discover_optional(adapter.client, "prompts")
                return tools, resources, prompts

            tools, resources, prompts = await self._run_sdk_operation(
                discover, failure="connect", name=name,
            )
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
            catalog_definitions = {
                "tools": sorted(definitions, key=lambda item: item["name"]),
                "resources": resources,
                "prompts": prompts,
            }
            record = {"config_digest": self._config_digest(name),
                      "catalog_digest": _digest(catalog_definitions),
                      "tool_schemas": [convert_to_openai_tool(entries[alias].tool) for alias in sorted(entries)],
                      "resources": resources, "prompts": prompts}
            if expected is not None and record != expected:
                raise MCPRecoveryError(f"mcp_catalog_mismatch: {name}; create a new session")
            catalog = MCPCatalog(record, entries)
            # No await between the global collision check and this atomic commit.
            self.catalogs[name] = catalog
            return catalog

    def saved_tools(self, state) -> list[dict[str, Any]]:
        """Return enabled, checkpointed model schemas without contacting servers."""
        self.validate_connections(state)
        schemas = []
        for name, record in state.get("mcp_connections", {}).items():
            tools = record.get("tool_schemas")
            if not isinstance(tools, list) or any(not isinstance(item, dict) for item in tools):
                raise MCPRecoveryError(f"mcp_catalog_missing: {name}; create a new session")
            schemas.extend(tools)
        return schemas

    def validate_connections(self, state) -> None:
        records = state.get("mcp_connections", {})
        # Validate *all* configuration before connecting to any of the saved servers.
        for name, record in records.items():
            if name not in self.config.servers or self._config_digest(name) != record["config_digest"]:
                raise MCPRecoveryError(f"mcp_config_mismatch: {name}; restore the configuration or create a new session")

    async def ensure_loaded(self, state) -> None:
        self.validate_connections(state)
        records = state.get("mcp_connections", {})
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

    def _enabled_adapter(self, name: str, state):
        self.validate_connections(state)
        if name not in self.config.servers:
            raise ToolException("mcp_unknown_server: choose one of " + ", ".join(self.config.servers))
        if name not in state.get("mcp_connections", {}):
            raise ToolException(f"mcp_server_not_connected: call connect_mcp for {name} first")
        return self._make_adapter(name)

    async def _client_operation(self, name: str, state, operation):
        # Match generated MCP tools: one bounded, short-lived connection per call.
        async def connected():
            adapter = self._enabled_adapter(name, state)
            async with adapter:
                return await operation(adapter.client)

        return await self._run_sdk_operation(connected)

    async def read_resource(
        self, name: str, uri: str, state, *, offset: int = 0, limit: int | None = None,
    ) -> str:
        try:
            validated_uri = AnyUrl(uri)
        except ValidationError:
            raise ToolException("mcp_invalid_resource_uri: provide a valid absolute resource URI") from None
        contents = await self._client_operation(name, state, lambda client: client.read_resource(validated_uri))
        return _paged_mcp_output(
            "resource", _resource_chunks(contents), len(contents), offset, limit, self.max_output_bytes,
        )

    async def get_prompt(
        self, name: str, prompt: str, arguments: dict[str, Any] | None, state,
        *, offset: int = 0, limit: int | None = None,
    ) -> str:
        result = await self._client_operation(name, state, lambda client: client.get_prompt(prompt, arguments))
        return _paged_mcp_output(
            "prompt", _prompt_chunks(result.messages), len(result.messages), offset, limit, self.max_output_bytes,
        )

    def entries(self, state) -> dict[str, MCPEntry]:
        # An in-flight connect can warm the cache before its Command reaches state.
        return {alias: entry for name in state.get("mcp_connections", {})
                for alias, entry in self.catalogs[name].entries.items()}

    def requires_capability_approval(
        self, server_id: str, kind: Literal["tools", "resources", "prompts"], raw_name: str,
    ) -> bool:
        definition = self.config.servers.get(server_id)
        if definition is None:
            return False
        return getattr(definition, kind).get(raw_name, "allow") == "confirm"

    def requires_approval(self, entry: MCPEntry) -> bool:
        return self.requires_capability_approval(entry.server_id, "tools", entry.raw_name)

    def check_sync(self, state) -> None:
        if state.get("mcp_connections"):
            for name, record in state["mcp_connections"].items():
                if name not in self.config.servers or self._config_digest(name) != record["config_digest"]:
                    raise MCPRecoveryError(f"mcp_config_mismatch: {name}; create a new session")
        if self.config.servers or state.get("mcp_connections"):
            raise MCPRecoveryError("mcp_async_required: use ainvoke for MCP-enabled sessions")
