import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx2
import pytest
import tomlkit
from fastmcp import Client
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.checkpoint.memory import InMemorySaver

from rs_harness.agents.main import build_agent
from rs_harness.application import ask, create_session, get_configured_mcp, get_pending_approvals, resume
from rs_harness.config import MCPConfig
from rs_harness.context import AgentContext
from rs_harness.mcp import MCPRegistry, cancel_elicitation
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.fixtures.mcp.server import events, make_server
from tests.fixtures.mcp.support import http_server, run_process, stdio_config
from tests.integration.test_permissions import call, decide, snapshot
from tests.models import ScriptedModel
from tests.unit.test_mcp import settings


def response(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def install_servers(monkeypatch, tmp_path, kinds=("docs", "deploy")):
    servers = {kind: make_server(kind, tmp_path / f"{kind}.jsonl") for kind in kinds}
    monkeypatch.setattr(MCPRegistry, "_make_adapter", lambda self, name: MCPAdapter(
        Client(servers[name], elicitation_handler=cancel_elicitation)))
    return servers


def successful_calls(log):
    return [item["tool"] for item in events(log) if item["event"] == "call"]


async def test_lazy_binding_and_untrusted_same_batch(config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path)
    config = config.model_copy(update={"mcp": settings(docs={})})
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "connect"), call("mcp__docs__search", {"query": "too soon"}, "guessed")),
        response(call("mcp__docs__search", {"query": "hooks"}, "search"), call("mcp__docs__get_version", {}, "version"),
                 call("mcp__docs__fake", {}, "fake")),
        AIMessage(content="done"),
    ])
    build_agent(config, workdir=str(tmp_path), model=model)
    assert not (tmp_path / "docs.jsonl").exists()
    result = await ask(config, session.session_id, "run", model=model)
    assert result.status == "succeeded", result.error
    first, second, last = [{tool.name for tool in batch} for batch in model.tool_history]
    assert "connect_mcp" in first and not any(name.startswith("mcp__") for name in first)
    assert second == last == first | {"mcp__docs__search", "mcp__docs__get_version"}
    search_tool = next(tool for tool in model.tool_history[1] if tool.name == "mcp__docs__search")
    assert search_tool.description == "Search project documentation, including agent hooks."
    messages = {message.tool_call_id: message for message in model.inputs[-1] if isinstance(message, ToolMessage)}
    assert messages["guessed"].status == messages["fake"].status == "error"
    assert messages["search"].artifact["structured_content"]["query"] == "hooks"
    assert messages["search"].name == "mcp__docs__search"
    assert messages["version"].content[0]["text"] == "docs-api-8"
    assert sorted(successful_calls(tmp_path / "docs.jsonl")) == ["get.version", "search"]
    systems = [str(batch[0].content) for batch in model.inputs]
    assert all("Available MCP servers" in text and "Available Skills" in text and "write_todos" in text for text in systems)
    assert all(text.count("## Available MCP servers") == 1 for text in systems)
    catalog = model.inputs[0][0].content[-1]["text"]
    assert "\n- docs\n" in catalog
    assert "http://127.0.0.1" not in catalog and "Fixture docs" not in catalog
    trajectory = session.trajectory_path.read_text()
    assert any(item["event"] == "tool_start" and item["payload"]["name"] == "mcp__docs__search"
               for item in (json.loads(line) for line in trajectory.splitlines()))
    assert "http://127.0.0.1" not in trajectory


@pytest.mark.parametrize("policies", [
    {},
    {"resources": {"fixture://catalog": "allow"}, "prompts": {"guide": "allow"}},
])
async def test_resource_and_prompt_bridges_follow_connection(config, tmp_path, monkeypatch, policies):
    install_servers(monkeypatch, tmp_path, ("docs",))
    config = config.model_copy(update={"mcp": settings(docs=policies)})
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "connect"),
                 call("read_mcp_resource", {"server": "docs", "uri": "fixture://catalog"}, "too_soon")),
        response(call("read_mcp_resource", {"server": "docs", "uri": "fixture://catalog"}, "resource"),
                 call("get_mcp_prompt", {"server": "docs", "name": "guide", "arguments": {"topic": "resources"}}, "prompt"),
                 call("read_mcp_resource", {"server": "docs", "uri": "not a uri"}, "invalid_uri")),
        AIMessage(content="done"),
    ])

    result = await ask(config, session.session_id, "read public contracts", model=model)

    assert result.status == "succeeded", result.error
    first, second, last = [{tool.name for tool in batch} for batch in model.tool_history]
    assert {"connect_mcp", "read_mcp_resource", "get_mcp_prompt"} <= first
    assert not any(name.startswith("mcp__") for name in first)
    assert second == last == first | {"mcp__docs__search", "mcp__docs__get_version"}
    messages = {message.tool_call_id: message for message in model.inputs[-1] if isinstance(message, ToolMessage)}
    assert messages["too_soon"].status == "error"
    assert "mcp_server_not_connected" in str(messages["too_soon"].content)
    assert messages["invalid_uri"].status == "error"
    assert "mcp_invalid_resource_uri" in str(messages["invalid_uri"].content)
    assert messages["resource"].status == messages["prompt"].status == "success"
    resource = json.loads(messages["resource"].content)
    prompt = json.loads(messages["prompt"].content)
    assert '{"schema_version":"fixture"' in resource["content"]
    assert "Read resources guidance before analysis." in prompt["content"]
    systems = [str(batch[0].content) for batch in model.inputs]
    assert "read_mcp_resource" not in systems[0] and "get_mcp_prompt" not in systems[0]
    assert all("read_mcp_resource" in text and "get_mcp_prompt" in text for text in systems[1:])
    assert "fixture://catalog" in systems[1] and "guide" in systems[1]


async def test_tools_only_catalog_does_not_advertise_resource_or_prompt_access(config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path, ("deploy",))
    config = config.model_copy(update={"mcp": settings(deploy={})})
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "deploy"}, "connect")),
        AIMessage(content="done"),
    ])

    result = await ask(config, session.session_id, "connect", model=model)

    assert result.status == "succeeded", result.error
    systems = [str(batch[0].content) for batch in model.inputs]
    assert all("read_mcp_resource" not in text and "get_mcp_prompt" not in text for text in systems)
    connected = str(model.inputs[-1][0].content)
    assert "resources: empty" in connected and "prompts: empty" in connected


@pytest.mark.parametrize("kind", ["resource", "prompt"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_resource_and_prompt_require_approval_before_access(config, tmp_path, monkeypatch, kind, decision):
    install_servers(monkeypatch, tmp_path, ("docs",))
    policies = ({"resources": {"fixture://catalog": "confirm"}} if kind == "resource"
                else {"prompts": {"guide": "confirm"}})
    config = config.model_copy(update={"mcp": settings(docs=policies)})
    session = create_session(config, workdir=str(tmp_path))
    tool_name = "read_mcp_resource" if kind == "resource" else "get_mcp_prompt"
    arguments = ({"server": "docs", "uri": "fixture://catalog"} if kind == "resource"
                 else {"server": "docs", "name": "guide"})
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "connect")),
        response(call(tool_name, arguments, "action")),
        AIMessage(content="done"),
    ])

    pending = await ask(config, session.session_id, "read", model=model)

    assert pending.status == "interrupted", pending.error
    assert [item["name"] for item in pending.interrupts[0]["action_requests"]] == [tool_name]
    event_name = "read_resource" if kind == "resource" else "get_prompt"
    log = tmp_path / "docs.jsonl"
    assert not any(item["event"] == event_name for item in events(log))
    result = await resume(config, session.session_id, decide(pending, {"type": decision}), model=model)
    assert result.status == "succeeded", result.error
    message = next(item for item in model.inputs[-1] if isinstance(item, ToolMessage) and item.tool_call_id == "action")
    assert message.status == ("success" if decision == "approve" else "error")
    assert sum(item["event"] == event_name for item in events(log)) == (1 if decision == "approve" else 0)


async def test_resource_and_prompt_approvals_have_independent_batch_decisions(config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path, ("docs",))
    config = config.model_copy(update={"mcp": settings(docs={
        "resources": {"fixture://catalog": "confirm"}, "prompts": {"guide": "confirm"},
    })})
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "connect")),
        response(call("read_mcp_resource", {"server": "docs", "uri": "fixture://catalog"}, "resource"),
                 call("get_mcp_prompt", {"server": "docs", "name": "guide"}, "prompt")),
        AIMessage(content="done"),
    ])

    pending = await ask(config, session.session_id, "read", model=model)

    assert pending.status == "interrupted", pending.error
    assert [item["name"] for item in pending.interrupts[0]["action_requests"]] == [
        "read_mcp_resource", "get_mcp_prompt",
    ]
    log = tmp_path / "docs.jsonl"
    assert not any(item["event"] in {"read_resource", "get_prompt"} for item in events(log))
    result = await resume(config, session.session_id, decide(
        pending, {"type": "approve"}, {"type": "reject"},
    ), model=model)
    assert result.status == "succeeded", result.error
    assert sum(item["event"] == "read_resource" for item in events(log)) == 1
    assert not any(item["event"] == "get_prompt" for item in events(log))
    messages = {item.tool_call_id: item for item in model.inputs[-1] if isinstance(item, ToolMessage)}
    assert messages["resource"].status == "success" and messages["prompt"].status == "error"


async def test_malformed_bridge_call_uses_tool_validation_before_approval(config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path, ("docs",))
    config = config.model_copy(update={"mcp": settings(docs={
        "resources": {"fixture://catalog": "confirm"},
    })})
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "connect")),
        response(call("read_mcp_resource", {"server": "docs"}, "malformed")),
        AIMessage(content="done"),
    ])

    result = await ask(config, session.session_id, "read", model=model)

    assert result.status == "succeeded", result.error
    message = next(item for item in model.inputs[-1] if isinstance(item, ToolMessage) and item.tool_call_id == "malformed")
    assert message.status == "error"
    assert not any(item["event"] == "read_resource" for item in events(tmp_path / "docs.jsonl"))


@pytest.mark.parametrize("decisions", [("approve", "reject"), ("reject", "approve"), ("reject", "reject")])
async def test_mixed_approval_and_explicit_confirm(config, tmp_path, monkeypatch, decisions):
    install_servers(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = config.model_copy(update={"mcp": settings(deploy={"tools": {"unlisted": "confirm"}})})
    session = create_session(config, workdir=str(workspace))
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "deploy"}, "connect")),
        response(call("mcp__deploy__unlisted", {}, "remote"),
                 call("write_file", {"path": "../outside", "content": "yes"}, "local"),
                 call("bash", {"command": "sudo forbidden"}, "deny")), AIMessage(content="done"),
    ])
    pending = await ask(config, session.session_id, "run", model=model)
    assert pending.status == "interrupted", pending.error
    assert [item["name"] for item in pending.interrupts[0]["action_requests"]] == ["mcp__deploy__unlisted", "write_file"]
    assert successful_calls(tmp_path / "deploy.jsonl") == [] and not (tmp_path / "outside").exists()
    before = events(tmp_path / "deploy.jsonl")
    status = await get_pending_approvals(config, session.session_id, model=model)
    assert status.interrupts == pending.interrupts and events(tmp_path / "deploy.jsonl") == before
    result = await resume(config, session.session_id, decide(pending, *({"type": decision} for decision in decisions)), model=model)
    assert result.status == "succeeded", result.error
    assert successful_calls(tmp_path / "deploy.jsonl") == (["unlisted"] if decisions[0] == "approve" else [])
    assert (tmp_path / "outside").exists() == (decisions[1] == "approve")
    denied = [item for item in model.inputs[-1] if isinstance(item, ToolMessage) and item.tool_call_id == "deny"]
    assert denied[0].status == "error"


async def test_multi_server_commands_and_repeated_connect(config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path)
    config = config.model_copy(update={"mcp": settings(docs={}, deploy={})})
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "a"), call("connect_mcp", {"name": "deploy"}, "b"),
                 call("connect_mcp", {"name": "docs"}, "duplicate")),
        response(call("connect_mcp", {"name": "docs"}, "again")), AIMessage(content="done"),
    ])
    session = create_session(config, workdir=str(tmp_path))
    assert (await ask(config, session.session_id, "connect", model=model)).status == "succeeded"
    assert set((await snapshot(config, session)).values["mcp_connections"]) == {"docs", "deploy"}
    assert sum(event["event"] == "start" for event in events(tmp_path / "docs.jsonl")) == 1
    again = next(item for item in model.inputs[-1] if isinstance(item, ToolMessage) and item.tool_call_id == "again")
    assert "already connected" in again.content


async def test_errors_outputs_and_images(config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path, ("all",))
    config = config.model_copy(update={"mcp": settings(all={}),
                                      "runtime": config.runtime.model_copy(update={"max_tool_output_bytes": 1024})})
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "all"}, "connect")),
        response(call("mcp__all__search", {}, "missing"), call("mcp__all__search", {"query": "x", "unknown": 1}, "extra"),
                 call("mcp__all__failure", {}, "failure"), call("mcp__all__large", {"size": 1000}, "large"),
                 call("mcp__all__picture", {}, "picture")), AIMessage(content="fixed"),
    ])
    result = await ask(config, session.session_id, "run", model=model)
    assert result.status == "succeeded", result.error
    messages = {m.tool_call_id: m for m in model.inputs[-1] if isinstance(m, ToolMessage)}
    assert all(messages[key].status == "error" for key in ("missing", "extra", "failure"))
    assert "search" not in successful_calls(tmp_path / "all.jsonl")
    assert messages["large"].status == "success" and messages["large"].artifact is None
    assert "output omitted" in messages["large"].content and len(messages["large"].content.encode()) <= 1024
    assert any(block["type"] == "image" for block in messages["picture"].content)
    assert "iVBOR" not in session.trajectory_path.read_text()


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_real_transports_sdk_aliases_pagination_and_lifecycle(tmp_path, transport):
    async def exercise(config, log):
        registry = MCPRegistry(config)
        assert not successful_calls(log)
        catalog = await registry.connect("docs", {})
        assert set(catalog.entries) == {"mcp__docs__search", "mcp__docs__get_version"}
        result = await catalog.entries["mcp__docs__get_version"].tool.ainvoke(call("mcp__docs__get_version", {}, "raw"))
        assert (result.name, result.tool_call_id, result.status) == ("mcp__docs__get_version", "raw", "success")
        assert result.content[0]["text"] == "docs-api-8"
        state = {"mcp_connections": {"docs": catalog.record}}
        resource = json.loads(await registry.read_resource("docs", "fixture://catalog", state))
        prompt = json.loads(await registry.get_prompt("docs", "guide", None, state))
        assert '{"schema_version":"fixture"' in resource["content"]
        assert "Read hooks guidance before analysis." in prompt["content"]
        assert any(item["uri"] == "fixture://catalog" for item in catalog.record["resources"]["items"])
        assert any(item["name"] == "guide" for item in catalog.record["prompts"]["items"])
        assert len(catalog.record["resources"]["items"]) == 6
        assert len(catalog.record["prompts"]["items"]) == 6
        assert successful_calls(log) == ["get.version"]
        assert sum(item["event"] == "read_resource" for item in events(log)) == 1
        assert sum(item["event"] == "get_prompt" for item in events(log)) == 1
        return events(log)

    if transport == "stdio":
        log = tmp_path / "stdio.jsonl"
        records = await exercise(stdio_config(log), log)
        pids = {item["pid"] for item in records if item["event"] == "start"}
        assert len(pids) >= 2  # discovery has exited before a fresh tool process starts
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    else:
        with http_server(tmp_path) as (url, log):
            await exercise(settings(docs={"url": url}), log)


@pytest.mark.parametrize("mode,tool_name", [("auto", "modern_input"), ("legacy", "elicitation")])
async def test_official_elicitation_is_cancelled(tmp_path, mode, tool_name):
    log = tmp_path / "elicitation.jsonl"
    client = Client(make_server("all", log), mode=mode, elicitation_handler=cancel_elicitation)
    tools = await MCPAdapter(client).list_tools()
    tool = next(item for item in tools if item.name == tool_name)
    result = await tool.ainvoke(call(tool_name, {}, "input"))
    assert result.status == "success" and "cancel" in result.content[0]["text"]
    assert [event["action"] for event in events(log) if event["event"] == "elicitation"] == ["cancel"]
    assert not client.is_connected()


async def test_noninteractive_continuation_over_real_http(tmp_path):
    with http_server(tmp_path, kind="all") as (url, log):
        registry = MCPRegistry(settings(all={"url": url}))
        catalog = await registry.connect("all", {})
        tool = catalog.entries["mcp__all__poll"].tool
        result = await tool.ainvoke(call(tool.name, {}, "poll"))
        assert result.status == "success" and result.content[0]["text"] == "completed"
        assert [event["step"] for event in events(log) if event["event"] == "poll_round"] == [0, 1, 2]
        assert successful_calls(log) == ["poll"]


@pytest.mark.parametrize("kind", ["timeout", "cancel"])
async def test_stdio_timeout_cancel_and_no_retries(tmp_path, kind):
    log = tmp_path / "stdio.jsonl"
    registry = MCPRegistry(stdio_config(log, kind="all", timeout=3))
    catalog = await registry.connect("all", {})
    failed = await catalog.entries["mcp__all__failure"].tool.ainvoke(call("mcp__all__failure", {}, "failure"))
    assert failed.status == "error"
    tool = catalog.entries["mcp__all__slow"].tool
    running = asyncio.create_task(tool.ainvoke(call(tool.name, {"seconds": 20}, "slow")))
    if kind == "cancel":
        async with asyncio.timeout(5):
            while "slow" not in successful_calls(log):
                await asyncio.sleep(0.02)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    else:
        result = await running
        assert result.status == "error" and "outcome may be unknown" in result.content
    assert successful_calls(log).count("slow") == 1
    for pid in {item["pid"] for item in events(log) if item["event"] == "start"}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def process_request(tmp_path, mcp, **kwargs):
    return run_process(tmp_path / "config.toml", workdir=str(tmp_path), mcp=mcp.model_dump(mode="json"), **kwargs)


def test_sqlite_new_process_resource_and_prompt_calls(config, tmp_path):
    log = tmp_path / "docs.jsonl"
    mcp = stdio_config(log)
    first = process_request(tmp_path, mcp, responses=[
        response(call("connect_mcp", {"name": "docs"}, "connect")).model_dump(),
        AIMessage(content="connected").model_dump(),
    ])["result"]
    assert first["status"] == "succeeded", first

    second = process_request(tmp_path, mcp, session_id=first["session_id"], responses=[
        response(call("read_mcp_resource", {"server": "docs", "uri": "fixture://catalog"}, "resource"),
                 call("get_mcp_prompt", {"server": "docs", "name": "guide"}, "prompt")).model_dump(),
        AIMessage(content="done").model_dump(),
    ])

    assert second["result"]["status"] == "succeeded", second["result"]
    messages = {item["tool_call_id"]: item for item in second["messages"] if item["type"] == "tool"}
    resource = json.loads(messages["resource"]["content"])
    prompt = json.loads(messages["prompt"]["content"])
    assert '{"schema_version":"fixture"' in resource["content"]
    assert "Read hooks guidance before analysis." in prompt["content"]
    assert sum(item["event"] == "read_resource" for item in events(log)) == 1
    assert sum(item["event"] == "get_prompt" for item in events(log)) == 1
    assert successful_calls(log) == []


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_sqlite_new_process_approval_query_and_continuation(config, tmp_path, decision):
    log = tmp_path / "deploy.jsonl"
    mcp = stdio_config(log, kind="deploy", tools={"trigger": "confirm"})
    pending = process_request(tmp_path, mcp, responses=[
        response(call("connect_mcp", {"name": "deploy"}, "connect")).model_dump(),
        response(call("mcp__deploy__status", {"service": "web"}, "status")).model_dump(),
        response(call("mcp__deploy__trigger", {"service": "web"}, "trigger")).model_dump(),
    ])["result"]
    assert pending["status"] == "interrupted", pending
    sid = pending["session_id"]
    before = events(log)
    queried = process_request(tmp_path, mcp, action="pending", session_id=sid)["result"]
    assert queried["interrupts"] == pending["interrupts"] and events(log) == before
    choices = {pending["interrupts"][0]["id"]: {"decisions": [{"type": decision}]}}
    resumed = process_request(tmp_path, mcp, action="resume", session_id=sid, decisions=choices)
    assert resumed["result"]["status"] == "succeeded", resumed
    assert resumed["result"]["interaction_id"] == pending["interaction_id"]
    assert successful_calls(log).count("trigger") == (1 if decision == "approve" else 0)
    assert successful_calls(log).count("status") == 1
    continued = process_request(tmp_path, mcp, session_id=sid)
    assert continued["result"]["status"] == "succeeded", continued
    assert "mcp__deploy__trigger" in continued["tool_history"][0]
    assert successful_calls(log).count("status") == 1  # graph rebuild does not replay tools
    record = continued["connections"]["deploy"]
    assert {"config_digest", "catalog_digest", "tool_schemas", "resources", "prompts"} <= set(record)
    assert set(record["resources"]) == set(record["prompts"]) == {"status", "items"}


@pytest.mark.parametrize("change", [
    "removed", "empty", "policy", "resource_policy", "prompt_policy", "target",
    "schema", "resource_schema", "prompt_schema", "unavailable",
])
def test_sqlite_recovery_refuses_changed_or_unavailable_server(config, tmp_path, change):
    log = tmp_path / "docs.jsonl"
    variant = tmp_path / "variant"
    variant.write_text("original")
    mcp = stdio_config(log, variant_file=variant, tools={"search": "confirm"},
                       resources={"fixture://catalog": "confirm"}, prompts={"guide": "confirm"})
    pending = process_request(tmp_path, mcp, responses=[
        response(call("connect_mcp", {"name": "docs"}, "connect")).model_dump(),
        response(call("mcp__docs__search", {"query": "hooks"}, "search")).model_dump(),
    ])["result"]
    assert pending["status"] == "interrupted", pending
    sid = pending["session_id"]
    choices = {pending["interrupts"][0]["id"]: {"decisions": [{"type": "approve"}]}}
    changed = mcp.model_dump(mode="json")
    if change in {"removed", "empty"}:
        changed = {} if change == "removed" else {"servers": {}}
    elif change == "policy":
        changed["servers"]["docs"]["tools"] = {}
    elif change == "resource_policy":
        changed["servers"]["docs"]["resources"] = {}
    elif change == "prompt_policy":
        changed["servers"]["docs"]["prompts"] = {}
    elif change == "target":
        changed["servers"]["docs"]["args"].append("--invalid")
    elif change == "schema":
        variant.write_text("changed")
    elif change in {"resource_schema", "prompt_schema"}:
        variant.write_text({"resource_schema": "changed_resource", "prompt_schema": "changed_prompt"}[change])
    elif change == "unavailable":
        # A stalled service keeps its routing config but cannot complete discovery.
        variant.write_text("slow_list")
    failed = process_request(tmp_path, MCPConfig.model_validate(changed), session_id=sid, action="resume", decisions=choices)["result"]
    assert failed["status"] == "failed" and "mcp_" in failed["error"]["message"], failed
    assert not successful_calls(log)
    still_pending = process_request(tmp_path, mcp, action="pending", session_id=sid)["result"]
    assert still_pending["interrupts"] == pending["interrupts"]
    variant.write_text("original")
    retried = process_request(tmp_path, mcp, session_id=sid, action="resume", decisions=choices)["result"]
    assert retried["status"] == "succeeded", retried
    assert successful_calls(log) == ["search"]


@pytest.mark.parametrize("error_kind", ["os", "http"])
@pytest.mark.parametrize("limit", [64, 65536])
async def test_transport_fault_is_redacted_before_callbacks(config, tmp_path, monkeypatch, error_kind, limit):
    from fastmcp import FastMCP
    from langchain_core.callbacks import AsyncCallbackHandler
    server = FastMCP("secrets")
    server.tool(lambda: "ok", name="safe")
    adapter = MCPAdapter(Client(server, elicitation_handler=cancel_elicitation))
    original_tools = await adapter.list_tools()

    async def fail(**kwargs):
        if error_kind == "http":
            request = httpx2.Request("POST", "https://private-target.example?token=FIXTURE_SECRET_TOKEN")
            raise httpx2.HTTPStatusError("FIXTURE_SECRET_TOKEN", request=request, response=httpx2.Response(401, request=request))
        raise OSError("Authorization: Bearer FIXTURE_SECRET_TOKEN https://private-target.example")

    original_tools[0] = original_tools[0].model_copy(update={"coroutine": fail})

    class BrokenAdapter:
        client = adapter.client

        async def __aenter__(self):
            await adapter.__aenter__()
            return self

        async def __aexit__(self, *args):
            return await adapter.__aexit__(*args)

        async def list_tools(self, **kwargs):
            return original_tools

    monkeypatch.setattr(MCPRegistry, "_make_adapter", lambda self, name: BrokenAdapter())
    config = config.model_copy(update={"mcp": settings(docs={}),
                                      "runtime": config.runtime.model_copy(update={"max_tool_output_bytes": limit})})
    model = ScriptedModel([response(call("connect_mcp", {"name": "docs"}, "connect")),
                           response(call("mcp__docs__safe", {}, "safe")), AIMessage(content="done")])
    session = create_session(config, workdir=str(tmp_path))
    class ToolResults(AsyncCallbackHandler):
        def __init__(self):
            self.outputs = []

        async def on_tool_end(self, output, **kwargs):
            if isinstance(output, ToolMessage):
                self.outputs.append(output)

    callback = ToolResults()
    result = await ask(config, session.session_id, "run", model=model, callbacks=[callback])
    assert result.status == "succeeded", result.error
    failed = next(item for item in model.inputs[-1] if isinstance(item, ToolMessage) and item.tool_call_id == "safe")
    assert failed.status == "error" and "outcome may be unknown" in failed.content
    assert len(failed.content.encode()) <= limit and "result received" not in failed.content
    assert (failed.name, failed.tool_call_id) == ("mcp__docs__safe", "safe")
    recorded = next(item for item in callback.outputs if item.tool_call_id == "safe")
    assert recorded == failed
    assert "FIXTURE_SECRET_TOKEN" not in session.trajectory_path.read_text()
    assert "private-target.example" not in session.trajectory_path.read_text()


async def test_failed_connect_can_correct_without_partial_state(config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path)
    config = config.model_copy(update={"mcp": settings(docs={})})
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "unknown"}, "bad")),
        response(call("connect_mcp", {"name": "docs"}, "good")), AIMessage(content="done"),
    ])
    session = create_session(config, workdir=str(tmp_path))
    assert (await ask(config, session.session_id, "run", model=model)).status == "succeeded"
    assert not any(item.name.startswith("mcp__") for item in model.tool_history[1])
    assert set((await snapshot(config, session)).values["mcp_connections"]) == {"docs"}


@pytest.mark.parametrize("failure", ["unknown_server", "connect_failure", "unknown_tool"])
async def test_mcp_routing_errors_respect_output_budget(config, tmp_path, monkeypatch, failure):
    config = config.model_copy(update={"mcp": settings(docs={}),
                                      "runtime": config.runtime.model_copy(update={"max_tool_output_bytes": 32})})
    if failure == "connect_failure":
        monkeypatch.setattr(MCPRegistry, "_make_adapter", lambda *args: (_ for _ in ()).throw(OSError("SECRET")))
        invocation = call("connect_mcp", {"name": "docs"}, "failed")
    elif failure == "unknown_server":
        invocation = call("connect_mcp", {"name": "unknown"}, "failed")
    else:
        invocation = call("mcp__docs__missing", {}, "failed")
    model = ScriptedModel([response(invocation), AIMessage(content="done")])
    session = create_session(config, workdir=str(tmp_path))
    result = await ask(config, session.session_id, "run", model=model)
    assert result.status == "succeeded", result.error
    failed = next(item for item in model.inputs[-1] if isinstance(item, ToolMessage) and item.tool_call_id == "failed")
    assert failed.status == "error" and len(failed.content.encode()) <= 32
    assert "SECRET" not in failed.content and "result received" not in failed.content
    assert not (await snapshot(config, session)).values.get("mcp_connections")


async def test_skill_todo_base_tools_and_child_isolation(skills_config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path)
    config = skills_config.model_copy(update={"mcp": settings(docs={})})
    (tmp_path / "input.txt").write_text("local text")
    child = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "child-connect"),
                 call("mcp__docs__search", {"query": "forged"}, "child-mcp"),
                 call("read_mcp_resource", {"server": "docs", "uri": "fixture://catalog"}, "child-resource"),
                 call("get_mcp_prompt", {"server": "docs", "name": "guide"}, "child-prompt")),
        AIMessage(content="child could not access MCP"),
    ])
    parent = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "connect")),
        response(call("load_skill", {"name": "fixture-skill"}, "skill"),
                 call("read_file", {"path": "input.txt"}, "local"),
                 call("delegate_task", {"agent_id": "worker", "query": "Check available tools"}, "child")),
        response(call("write_todos", {"todos": [{"content": "Use MCP", "status": "in_progress"}]}, "todo")),
        response(call("mcp__docs__search", {"query": "hooks"}, "search")),
        response(call("write_todos", {"todos": [{"content": "Use MCP", "status": "completed"}]}, "done-todo")),
        AIMessage(content="done"),
    ])
    graph = build_agent(config, workdir=str(tmp_path), model=parent, subagent_model=child, checkpointer=InMemorySaver())
    result = await graph.ainvoke({"messages": [HumanMessage("run")]}, {"configurable": {"thread_id": "compose"}}, context=AgentContext(str(tmp_path)))
    assert result["messages"][-1].content == "done"
    assert result["todos"][0]["status"] == "completed"
    assert successful_calls(tmp_path / "docs.jsonl") == ["search"]
    forbidden = {"connect_mcp", "read_mcp_resource", "get_mcp_prompt"}
    assert all(not any(item.name in forbidden or item.name.startswith("mcp__") for item in batch) for batch in child.tool_history)
    assert {m.tool_call_id for m in child.inputs[-1] if isinstance(m, ToolMessage) and m.status == "error"} == {
        "child-connect", "child-mcp", "child-resource", "child-prompt",
    }
    assert not any(m.tool_call_id.startswith("child-") for m in result["messages"] if isinstance(m, ToolMessage))
    assert not any("Available MCP servers" in str(batch[0].content) for batch in child.inputs)


async def test_mcp_does_not_reset_model_budget(config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path)
    config = config.model_copy(update={"mcp": settings(docs={}),
                                      "runtime": config.runtime.model_copy(update={"max_model_calls": 2})})
    model = ScriptedModel([response(call("connect_mcp", {"name": "docs"}, "c")),
                           response(call("mcp__docs__search", {"query": "hooks"}, "s")), AIMessage(content="unreachable")])
    session = create_session(config, workdir=str(tmp_path))
    result = await ask(config, session.session_id, "run", model=model)
    assert result.status == "failed" and result.error["type"] == "ModelCallLimitExceededError"
    assert model.call_count == 2 and successful_calls(tmp_path / "docs.jsonl") == ["search"]


def test_sqlite_resume_directly_at_tool_node(config, tmp_path):
    log = tmp_path / "docs.jsonl"
    mcp = stdio_config(log)
    first = process_request(tmp_path, mcp, responses=[response(call("connect_mcp", {"name": "docs"}, "c")).model_dump(),
                                                    {"content": "connected"}])
    sid = first["result"]["session_id"]
    paused = process_request(tmp_path, mcp, session_id=sid, action="tool_pause",
                             responses=[response(call("mcp__docs__search", {"query": "hooks"}, "search")).model_dump()])
    assert paused["result"]["status"] == "succeeded" and successful_calls(log) == []
    refused = process_request(tmp_path, MCPConfig(), session_id=sid, action="tool_resume")
    assert refused["result"]["status"] == "failed" and "mcp_config_mismatch" in refused["result"]["error"]
    assert successful_calls(log) == []
    resumed = process_request(tmp_path, mcp, session_id=sid, action="tool_resume")
    assert resumed["result"]["status"] == "succeeded", resumed
    assert successful_calls(log) == ["search"]


@pytest.mark.parametrize("approval", [False, True])
def test_cli_mcp_json_is_one_result(config, tmp_path, monkeypatch, approval):
    from typer.testing import CliRunner
    from rs_harness import application, cli

    log = tmp_path / "docs.jsonl"
    configured = config.model_copy(update={"mcp": stdio_config(log, tools={"search": "confirm"} if approval else {})})
    model = ScriptedModel([response(call("connect_mcp", {"name": "docs"}, "c")),
                           response(call("mcp__docs__search", {"query": "hooks"}, "s")), AIMessage(content="done")])
    monkeypatch.setattr(cli, "_load", lambda path: configured)

    async def scripted_ask(config, session_id, text, **kwargs):
        return await application.ask(config, session_id, text, model=model, **kwargs)

    monkeypatch.setattr(cli, "ask", scripted_ask)
    result = CliRunner().invoke(cli.app, ["ask", "run", "--json", "--workdir", str(tmp_path)])
    assert result.exit_code == (2 if approval else 0), result.output
    data = json.loads(result.stdout)
    assert data["status"] == ("interrupted" if approval else "succeeded")
    assert successful_calls(log) == ([] if approval else ["search"])
    assert result.stderr == ""


async def test_disconnected_http_call_becomes_bounded_error(tmp_path):
    with http_server(tmp_path) as (url, log):
        registry = MCPRegistry(settings(docs={"url": url}))
        catalog = await registry.connect("docs", {})
    tool = catalog.entries["mcp__docs__search"].tool
    result = await tool.ainvoke(call(tool.name, {"query": "hooks"}, "down"))
    assert result.status == "error" and "outcome may be unknown" in result.content
    assert url not in result.content
    assert successful_calls(log) == []


async def test_configured_mcp_query_discovers_all_servers_without_session_or_model(config, tmp_path, monkeypatch):
    from rs_harness import application

    install_servers(monkeypatch, tmp_path)
    config = config.model_copy(update={"mcp": settings(docs={"tools": {"search": "confirm"}}, deploy={}),
                                      "model": config.model.model_copy(update={"api_key": ""})})

    def unexpected(*args, **kwargs):
        pytest.fail("MCP inspection must not read/create a session, checkpoint or model")

    monkeypatch.setattr(application, "build_agent", unexpected)
    monkeypatch.setattr(application, "open_checkpointer", unexpected)
    monkeypatch.setattr(application.SessionCatalog, "open", unexpected)
    monkeypatch.setattr(application.SessionCatalog, "create", unexpected)
    listed = await get_configured_mcp(config)
    assert [item["name"] for item in listed] == ["deploy", "docs"]
    tools = {item["raw_name"]: item for item in listed[1]["tools"]}
    assert tools["get.version"]["name"] == "mcp__docs__get_version"
    assert tools["get.version"]["permission"] == "allow"
    assert tools["search"]["permission"] == "confirm"
    assert tools["search"]["description"] == "Search project documentation, including agent hooks."
    assert tools["search"]["input_schema"]["required"] == ["query"]
    assert "127.0.0.1" not in json.dumps(listed)
    assert not config.runtime.sessions_dir.exists()
    for name in ("docs", "deploy"):
        assert any(item["event"] == "list" for item in events(tmp_path / f"{name}.jsonl"))
        assert successful_calls(tmp_path / f"{name}.jsonl") == []


async def test_configured_mcp_query_does_not_change_existing_pending_session(config, tmp_path, monkeypatch):
    from rs_harness import application

    install_servers(monkeypatch, tmp_path)
    config = config.model_copy(update={"mcp": settings(docs={"tools": {"search": "confirm"}}, deploy={})})
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "connect")),
        response(call("mcp__docs__search", {"query": "hooks"}, "pending")),
    ])
    pending = await ask(config, session.session_id, "connect", model=model)
    assert pending.status == "interrupted"
    run_config = {"configurable": {"thread_id": session.session_id, "checkpoint_ns": ""}}
    async with open_checkpointer(config) as saver:
        before = await saver.aget_tuple(run_config)
    trajectory, metadata = session.trajectory_path.read_bytes(), session.state_path.read_bytes()

    def unexpected(*args):
        pytest.fail("Listing configured MCP servers must not consult the session checkpoint")

    monkeypatch.setattr(application, "open_checkpointer", unexpected)
    assert [item["name"] for item in await get_configured_mcp(config)] == ["deploy", "docs"]
    async with open_checkpointer(config) as saver:
        after = await saver.aget_tuple(run_config)
    assert before == after
    assert set(after.checkpoint["channel_values"]["mcp_connections"]) == {"docs"}
    assert session.trajectory_path.read_bytes() == trajectory and session.state_path.read_bytes() == metadata
    assert model.call_count == 2
    assert successful_calls(tmp_path / "docs.jsonl") == successful_calls(tmp_path / "deploy.jsonl") == []


async def test_configured_mcp_query_uses_current_catalog_and_empty_config(config, tmp_path, monkeypatch):
    servers = install_servers(monkeypatch, tmp_path)
    config = config.model_copy(update={"mcp": settings(docs={})})
    first = await get_configured_mcp(config)
    search = next(item for item in first[0]["tools"] if item["raw_name"] == "search")
    assert search["input_schema"]["required"] == ["query"]
    servers["docs"] = make_server("docs", tmp_path / "docs.jsonl", variant="changed")
    second = await get_configured_mcp(config)
    search = next(item for item in second[0]["tools"] if item["raw_name"] == "search")
    assert search["input_schema"]["required"] == ["query", "required_scope"]
    assert await get_configured_mcp(config.model_copy(update={"mcp": MCPConfig()})) == []
    assert not config.runtime.sessions_dir.exists()


async def test_configured_mcp_query_reports_discovery_errors(config, tmp_path, monkeypatch):
    config = config.model_copy(update={"mcp": settings(docs={})})
    monkeypatch.setattr(MCPRegistry, "_make_adapter", lambda *args: (_ for _ in ()).throw(OSError("PRIVATE_TRANSPORT_DETAIL")))
    with pytest.raises(ToolException, match="mcp_connect_failed") as error:
        await get_configured_mcp(config)
    assert "PRIVATE_TRANSPORT_DETAIL" not in str(error.value)
    assert not config.runtime.sessions_dir.exists()


@pytest.mark.parametrize("kind", ["docs", "empty"])
def test_list_mcp_and_slash_mcp_read_real_stdio_without_session(config, tmp_path, kind):
    log = tmp_path / "mcp.jsonl"
    mcp = stdio_config(log, kind=kind)
    config_path = tmp_path / "config.toml"
    document = tomlkit.parse(config_path.read_text())
    document["mcp"] = mcp.model_dump(mode="json")
    document["model"]["api_key"] = ""
    config_path.write_text(tomlkit.dumps(document))
    executable = str(Path(sys.executable).with_name("rsh"))
    options = ["--config", str(config_path)]
    listed = subprocess.run([executable, "list-mcp", *options], capture_output=True, text=True, timeout=30)
    interactive = subprocess.run([executable, *options], input="/mcp\n/quit\n", capture_output=True, text=True, timeout=30)
    assert listed.returncode == interactive.returncode == 0, (listed.stderr, interactive.stderr)
    assert listed.stdout.strip() in interactive.stdout
    assert "MCP servers" in listed.stdout
    assert f"1 server · {2 if kind == 'docs' else 0} tools" in listed.stdout
    assert f"{kind} · stdio · {2 if kind == 'docs' else 0} tools" in listed.stdout
    assert listed.stdout.count("╭") == listed.stdout.count("╰") == 1
    if kind == "docs":
        assert "get.version" in listed.stdout and "search" in listed.stdout
        assert listed.stdout.count("allow") == 2
        normalized = "".join(listed.stdout.replace("│", "").split())
        assert "GetthedocumentationAPIversion." in normalized
        assert "Searchprojectdocumentation,includingagenthooks." in normalized
        assert "mcp__" not in listed.stdout and "MCP name:" not in listed.stdout
        assert "Input schema" not in listed.stdout and '"query"' not in listed.stdout
    else:
        assert "No tools exposed." in listed.stdout
    assert "\x1b" not in listed.stdout
    assert successful_calls(log) == [] and listed.stderr == interactive.stderr == ""
    assert sum(item["event"] == "start" for item in events(log)) == 2
    assert not config.runtime.sessions_dir.exists()


async def test_configured_mcp_query_real_http_without_session(config, tmp_path):
    with http_server(tmp_path) as (url, log):
        config = config.model_copy(update={"mcp": settings(docs={"url": url})})
        listed = await get_configured_mcp(config)
        assert [item["name"] for item in listed] == ["docs"]
        assert {item["raw_name"] for item in listed[0]["tools"]} == {"search", "get.version"}
        assert any(item["event"] == "list" for item in events(log))
        assert successful_calls(log) == []
        assert not config.runtime.sessions_dir.exists()


def test_interactive_mcp_lists_configuration_without_starting_conversation(config, tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from rs_harness import cli

    install_servers(monkeypatch, tmp_path)
    config = config.model_copy(update={"mcp": settings(docs={}, deploy={})})

    async def unexpected(*args, **kwargs):
        pytest.fail("/mcp must not start a conversation or query session state")

    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "ask", unexpected)
    monkeypatch.setattr(cli, "get_pending_approvals", unexpected)
    result = CliRunner().invoke(cli.app, [], input="/mcp\n/mcp\n/quit\n")
    assert result.exit_code == 0, result.output
    assert result.stdout.count("MCP servers") == 2
    assert result.stdout.count("2 servers · 5 tools") == 2
    assert all(name in result.stdout for name in ("get.version", "trigger", "unlisted"))
    assert result.stdout.count("allow") == 10
    assert not config.runtime.sessions_dir.exists()
    assert successful_calls(tmp_path / "docs.jsonl") == successful_calls(tmp_path / "deploy.jsonl") == []
