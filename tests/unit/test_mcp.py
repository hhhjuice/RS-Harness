import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastmcp import Client, Context, FastMCP
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.tools import ToolException
from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt
from mcp.types import InputRequiredResult
from pydantic import ValidationError

from rs_harness.agents.mcp import MCPMiddleware, merge_connections
from rs_harness.config import Config, MCPConfig
from rs_harness.mcp import MCPRecoveryError, MCPRegistry, bounded_error, bounded_output, cancel_elicitation, exposed_name
from tests.fixtures.mcp.server import events, make_server
from tests.models import ScriptedModel


def settings(**servers):
    return MCPConfig.model_validate({"servers": {name: {"transport": "http",
                                                       "url": "http://127.0.0.1:8000/mcp", **definition}
                                                for name, definition in servers.items()}})


def in_memory_registry(monkeypatch, config, servers, **kwargs):
    registry = MCPRegistry(config, **kwargs)
    monkeypatch.setattr(registry, "_make_adapter", lambda name: MCPAdapter(Client(servers[name], elicitation_handler=cancel_elicitation)))
    return registry


def test_config_paths_and_private_summary(config, tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    with path.open("a") as handle:
        handle.write('''
[mcp.servers.local]
transport = "stdio"
command = "python"
args = ["relative.py", "--password-reference"]
env_from_env = { TOKEN = "HOST_TOKEN" }
[mcp.servers.remote]
transport = "http"
url = "https://private.example/mcp?private=target"
bearer_token_env = "HOST_TOKEN"
''')
    monkeypatch.chdir(tmp_path.parent)
    loaded = Config.load(path)
    assert loaded.mcp.servers["local"].cwd == tmp_path
    assert loaded.mcp.servers["local"].args == ["relative.py", "--password-reference"]
    assert all("description" not in item.model_dump() for item in loaded.mcp.servers.values())
    assert loaded.summary()["mcp"] == {
        "local": {"transport": "stdio", "policy_count": 0},
        "remote": {"transport": "http", "policy_count": 0},
    }
    summary = json.dumps(loaded.summary())
    assert "private.example" not in summary and "HOST_TOKEN" not in summary and "relative.py" not in summary
    path.write_text(path.read_text().replace('command = "python"', 'command = "python"\ncwd = "subdir"'))
    assert Config.load(path).mcp.servers["local"].cwd == tmp_path / "subdir"


@pytest.mark.parametrize("change", [
    {"transport": "sse"}, {"url": "file:///tmp/server.py"}, {"command": "python"},
    {"tools": {"x": "deny"}}, {"description": "custom server description"},
    {"unknown": True}, {"bearer_token_env": ""},
])
def test_invalid_http_config(change):
    with pytest.raises(ValidationError):
        settings(docs=change)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), "30", True])
def test_invalid_timeout(value):
    with pytest.raises(ValidationError):
        MCPConfig(timeout_s=value)


@pytest.mark.parametrize("server", [
    {"transport": "stdio"},
    {"transport": "stdio", "command": " "},
    {"transport": "stdio", "command": "python", "url": "https://x.test"},
    {"transport": "stdio", "command": "python", "env_from_env": {"": "X"}},
    {"transport": "stdio", "command": "python", "description": "custom server description"},
])
def test_invalid_stdio_config(server):
    with pytest.raises(ValidationError):
        MCPConfig(servers={"x": server})


@pytest.mark.parametrize("names", [[""], ["x\ny"], ["a.b", "a/b"]])
def test_invalid_server_ids(names):
    with pytest.raises(ValidationError):
        settings(**{name: {} for name in names})


def test_alias_boundaries_and_state_reducer():
    assert exposed_name("a.b", "x/y") == "mcp__a_b__x_y"
    assert len(exposed_name("s", "x" * 56)) == 64
    for raw in ("", "x" * 57):
        with pytest.raises(ToolException):
            exposed_name("s", raw)
    record = {"config_digest": "a", "catalog_digest": "b"}
    assert merge_connections({"x": record}, {"x": record, "y": record}) == {"x": record, "y": record}
    with pytest.raises(MCPRecoveryError):
        merge_connections({"x": record}, {"x": {"config_digest": "other"}})


async def test_registry_atomic_conflicts(monkeypatch):
    server = FastMCP("collisions")
    server.tool(lambda: "x", name="a.b")
    server.tool(lambda: "y", name="a/b")
    registry = in_memory_registry(monkeypatch, settings(s={}), {"s": server})
    with pytest.raises(ToolException, match="mcp_name_conflict"):
        await registry.connect("s", {})
    assert registry.catalogs == {}


async def test_local_and_delimiter_conflicts(monkeypatch):
    first, second = FastMCP("one"), FastMCP("two")
    first.tool(lambda: "x", name="c")
    second.tool(lambda: "x", name="b__c")
    registry = in_memory_registry(monkeypatch, settings(a__b={}, a={}), {"a__b": first, "a": second})
    await registry.connect("a__b", {})
    with pytest.raises(ToolException, match="mcp_name_conflict"):
        await registry.connect("a", {})
    assert set(registry.catalogs) == {"a__b"}
    registry = in_memory_registry(monkeypatch, settings(a={}), {"a": first}, local_names=["mcp__a__c"])
    with pytest.raises(ToolException, match="mcp_name_conflict"):
        await registry.connect("a", {})
    assert not registry.catalogs


async def test_empty_parallel_duplicate_and_same_raw_names(monkeypatch, tmp_path):
    log = tmp_path / "calls.jsonl"
    server = make_server("docs", log)
    registry = in_memory_registry(monkeypatch, settings(a={}, b={}, empty={}),
                                  {"a": server, "b": server, "empty": FastMCP("empty")})
    first, duplicate = await asyncio.gather(registry.connect("a", {}), registry.connect("a", {}))
    assert first is duplicate
    assert sum(item["event"] == "start" for item in events(log)) == 1
    second, empty = await asyncio.gather(registry.connect("b", {}), registry.connect("empty", {}))
    state = {"mcp_connections": {"a": first.record, "b": second.record, "empty": empty.record}}
    assert set(registry.entries(state)) == {"mcp__a__search", "mcp__a__get_version", "mcp__b__search", "mcp__b__get_version"}
    assert registry.entries({}) == {}  # warmed cache does not authorize a tool
    assert empty.entries == {}
    with pytest.raises(ToolException, match="mcp_unknown_server"):
        await registry.connect("https://attacker.example", {})


async def test_missing_secret_and_rotation(monkeypatch):
    config = settings(docs={"bearer_token_env": "RSH_MISSING_TEST_TOKEN"})
    registry = MCPRegistry(config)
    monkeypatch.delenv("RSH_MISSING_TEST_TOKEN", raising=False)
    with pytest.raises(ToolException, match="RSH_MISSING_TEST_TOKEN"):
        await registry.connect("docs", {})
    monkeypatch.setenv("RSH_MISSING_TEST_TOKEN", "secret-one")
    digest = registry._config_digest("docs")
    monkeypatch.setenv("RSH_MISSING_TEST_TOKEN", "secret-two")
    assert registry._config_digest("docs") == digest
    assert "secret" not in json.dumps(config.model_dump(mode="json"))


@pytest.mark.parametrize("change", ["removed", "policy", "schema", "unavailable"])
async def test_restore_fails_before_execution(monkeypatch, tmp_path, change):
    log = tmp_path / "events"
    config = settings(docs={})
    registry = in_memory_registry(monkeypatch, config, {"docs": make_server("docs", log)})
    saved = await registry.connect("docs", {})
    state = {"mcp_connections": {"docs": saved.record}}
    if change == "removed":
        config = MCPConfig()
    if change == "policy":
        config = settings(docs={"tools": {"search": "confirm"}})
    restored = in_memory_registry(monkeypatch, config, {"docs": make_server("docs", log, variant="changed" if change == "schema" else "original")})
    if change == "unavailable":
        monkeypatch.setattr(restored, "_make_adapter", lambda name: (_ for _ in ()).throw(OSError("SECRET")))
    with pytest.raises(MCPRecoveryError) as error:
        await restored.ensure_loaded(state)
    assert "SECRET" not in str(error.value)
    assert not restored.catalogs
    assert not any(item["event"] == "call" for item in events(log))


@pytest.mark.parametrize("limit", [1, 40, 100])
def test_bounded_output(limit):
    content, artifact = bounded_output([{"type": "text", "text": "文" * 100}], {"data": "x" * 100}, limit)
    assert len(content.encode()) <= limit and artifact is None


@pytest.mark.parametrize("limit", [1, 4, 64])
def test_error_budget_preserves_utf8_and_error_meaning(limit):
    message = "失败：结果未知" * 10
    bounded = bounded_error(ToolException(message), limit)
    assert len(bounded.encode()) <= limit
    assert message.startswith(bounded) and "result received" not in bounded


async def test_programming_errors_and_cancellation_propagate(monkeypatch):
    registry = MCPRegistry(settings(docs={}))
    for error in (ValueError("bug"), asyncio.CancelledError(), GraphInterrupt(())):
        monkeypatch.setattr(registry, "_make_adapter", lambda name: (_ for _ in ()).throw(error))
        with pytest.raises(type(error)):
            await registry.connect("docs", {})
        assert not registry.catalogs


def test_sync_empty_passthrough_and_explicit_mcp_error():
    for config, expected in [(MCPConfig(), None), (settings(docs={}), "mcp_async_required")]:
        agent = create_agent(ScriptedModel([AIMessage(content="done")]), middleware=[MCPMiddleware(MCPRegistry(config))])
        if expected:
            with pytest.raises(MCPRecoveryError, match=expected):
                agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
        else:
            assert agent.invoke({"messages": [{"role": "user", "content": "hello"}]})["messages"][-1].content == "done"
            with pytest.raises(MCPRecoveryError, match="mcp_config_mismatch"):
                agent.invoke({"messages": [{"role": "user", "content": "hello"}],
                              "mcp_connections": {"docs": {"config_digest": "gone", "catalog_digest": "gone"}}})


@pytest.mark.parametrize("rounds", [2, None])
async def test_input_required_state_only_uses_sdk_continuation_limit(monkeypatch, rounds):
    server = FastMCP("repeat")
    steps = []

    @server.tool(output_schema=None)
    def waiting(ctx: Context):
        step = int(ctx.request_state or "0")
        steps.append(step)
        if rounds is None or step < rounds:
            return InputRequiredResult(requestState=str(step + 1))
        return "completed"

    # Keep the production Client construction, including its continuation settings.
    monkeypatch.setattr("rs_harness.mcp.StreamableHttpTransport", lambda *args, **kwargs: server)
    registry = MCPRegistry(settings(docs={}))
    catalog = await registry.connect("docs", {})
    result = await catalog.entries["mcp__docs__waiting"].tool.ainvoke({
        "name": "mcp__docs__waiting", "args": {}, "id": "input", "type": "tool_call"})
    if rounds is not None:
        assert result.status == "success" and result.content[0]["text"] == "completed"
        assert steps == [0, 1, 2]
    else:
        assert result.status == "error" and "mcp_input_required_limit" in result.content
        assert "elicitation" not in result.content
        assert len(steps) == Client(server).input_required_max_rounds + 1


@pytest.mark.parametrize("policy,requires_approval", [(None, False), ("allow", False), ("confirm", True)])
async def test_official_open_object_schema_and_metadata_are_preserved(monkeypatch, policy, requires_approval):
    from langchain_core.utils.function_calling import convert_to_openai_tool
    server = FastMCP("schema")

    @server.tool(meta={"origin": "pretend-allowed"})
    def mapping(payload: dict) -> dict:
        """Echo a JSON object."""
        return payload

    definition = {} if policy is None else {"tools": {"mapping": policy}}
    registry = in_memory_registry(monkeypatch, settings(docs=definition), {"docs": server})
    catalog = await registry.connect("docs", {})
    entry = catalog.entries["mcp__docs__mapping"]
    assert entry.tool.description == "Echo a JSON object."
    schema = convert_to_openai_tool(entry.tool)["function"]["parameters"]
    assert schema["properties"]["payload"]["additionalProperties"] is True
    assert registry.requires_approval(entry) is requires_approval  # server metadata cannot override host policy
    output = await entry.tool.ainvoke({"name": entry.tool.name, "args": {"payload": {"key": [1, "two"]}}, "id": "x", "type": "tool_call"})
    assert output.artifact["structured_content"] == {"key": [1, "two"]}


async def test_http_auth_failure_is_redacted_and_oauth_token_is_literal(monkeypatch):
    received = []

    class Unauthorized(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.headers.get("Authorization"))
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(401)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Unauthorized)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    monkeypatch.setenv("RSH_MCP_TEST_TOKEN", "oauth")
    config = settings(docs={"url": f"http://127.0.0.1:{server.server_port}/private?token=FIXTURE_SECRET",
                            "bearer_token_env": "RSH_MCP_TEST_TOKEN"})
    try:
        registry = MCPRegistry(config)
        with pytest.raises(ToolException, match="mcp_connect_failed") as exc:
            await registry.connect("docs", {})
        assert "FIXTURE_SECRET" not in str(exc.value)
        assert received and set(received) == {"Bearer oauth"}
        assert not registry.catalogs
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)
