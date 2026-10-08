import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from rs_harness.agents.main import build_agent_components
from rs_harness.application import ask, create_session, get_context_usage
from rs_harness.context_usage import estimate_categories, model_key
from rs_harness.mcp import MCPRegistry
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.integration.test_mcp import install_servers, response
from tests.integration.test_permissions import call
from tests.integration.test_skills import put_skill
from tests.models import ScriptedModel
from tests.unit.test_mcp import settings


async def saved_state(config, sid):
    async with open_checkpointer(config, read_only=True) as saver:
        return await saver.aget_tuple({"configurable": {"thread_id": sid, "checkpoint_ns": ""}})


async def test_empty_session_counts_base_context_without_creating_checkpoint(config, tmp_path, monkeypatch):
    session = create_session(config, workdir=str(tmp_path))
    def forbidden(*args, **kwargs):
        pytest.fail("Context inspection must not initialize a model or connect MCP")
    monkeypatch.setattr("langchain_openai.ChatOpenAI.__init__", forbidden)
    monkeypatch.setattr(MCPRegistry, "_make_adapter", forbidden)
    usage = await get_context_usage(config, session.session_id)
    categories = {item.key: item.tokens for item in usage.categories}
    assert categories["system"] > 0 and categories["tool_schemas"] > 0 and categories["skills"] > 0
    assert categories["user"] == categories["assistant"] == categories["tool_results"] == 0
    assert usage.context_window is usage.usage_percent is usage.remaining_tokens is None
    assert not config.runtime.checkpoint_file.exists()


async def test_projection_matches_real_request_and_includes_latest_answer(config, tmp_path, monkeypatch):
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="最新回复 " * 100)])
    result = await ask(config, session.session_id, "Question", model=model)
    assert result.status == "succeeded", result.error
    before = await saved_state(config, session.session_id)
    files = [session.path / "session.json", session.trajectory_path, config.runtime.checkpoint_file]
    original = [path.read_bytes() for path in files]
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected model/client construction")
    monkeypatch.setattr("langchain_openai.ChatOpenAI.__init__", forbidden)
    usage = await get_context_usage(config, session.session_id)
    assert [path.read_bytes() for path in files] == original
    assert before == await saved_state(config, session.session_id)
    assert model.call_count == 1
    request = build_agent_components(config, workdir=str(tmp_path)).context_request(before.checkpoint["channel_values"])
    assert request.system_message.content == model.inputs[0][0].content
    assert [convert_to_openai_tool(tool) for tool in request.tools] == [convert_to_openai_tool(tool) for tool in model.tool_history[0]]
    expected = count_tokens_approximately([request.system_message, *request.messages],
                                          tools=[convert_to_openai_tool(tool) for tool in request.tools])
    assert usage.used_tokens == expected
    assert next(item.tokens for item in usage.categories if item.key == "assistant") == count_tokens_approximately([request.messages[-1]])


async def test_calibration_uses_last_root_input_and_clears_when_usage_missing(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        AIMessage(content="First", usage_metadata={"input_tokens": 7000, "output_tokens": 100000, "total_tokens": 107000}),
        AIMessage(content="Second", usage_metadata={"input_tokens": 12000, "output_tokens": 10, "total_tokens": 12010}),
        AIMessage(content="No usage"),
    ])
    await ask(config, session.session_id, "First question", model=model)
    usage = await get_context_usage(config, session.session_id)
    assert usage.estimation_method == "calibrated"
    assert 7000 < usage.used_tokens < 8000  # excludes output usage, includes only actual reply text
    await ask(config, session.session_id, "Second question", model=model)
    state = (await saved_state(config, session.session_id)).checkpoint["channel_values"]
    sample = state["context_calibration"]
    assert sample["input_tokens"] == 12000 and sample["model_key"] == model_key(config)
    assert sample["estimated_input_tokens"] == sum(estimate_categories(
        model.inputs[-1][0], model.inputs[-1][1:], model.tool_history[-1],
        skills_prompt=build_agent_components(config, workdir=str(tmp_path)).skills_prompt,
    ).values())
    await ask(config, session.session_id, "No usage this time", model=model)
    assert (await get_context_usage(config, session.session_id)).estimation_method == "approximate"


async def test_responses_tool_call_calibration_and_checkpoint_usage(config, tmp_path):
    from openai.types.responses import Response, ResponseFunctionToolCall
    from langchain_openai.chat_models.base import _construct_lc_result_from_responses_api

    # Use the installed provider converter so this covers its real dual representation.
    arguments = {"path": "note.txt", "content": "Tool input " * 1000}
    response = Response.model_construct(
        id="resp_fixture", model="gpt-5-pro", error=None, usage=None,
        output=[ResponseFunctionToolCall(
            type="function_call", name="write_file", arguments=json.dumps(arguments),
            call_id="call_fixture", id="fc_fixture", status="completed",
        )],
    )
    tool_call = _construct_lc_result_from_responses_api(response).generations[0].message
    reply = AIMessage(content="Done", usage_metadata={"input_tokens": 12000, "output_tokens": 1, "total_tokens": 12001})
    model = ScriptedModel([tool_call, reply])
    session = create_session(config, workdir=str(tmp_path))

    result = await ask(config, session.session_id, "Write the note", model=model)

    assert result.status == "succeeded", result.error
    assert (tmp_path / "note.txt").read_text() == arguments["content"]
    expected_input = count_tokens_approximately(
        model.inputs[-1], tools=[convert_to_openai_tool(tool) for tool in model.tool_history[-1]],
    )
    state = (await saved_state(config, session.session_id)).checkpoint["channel_values"]
    assert state["context_calibration"]["estimated_input_tokens"] == expected_input
    usage = await get_context_usage(config, session.session_id)
    expected_assistant = round(count_tokens_approximately([tool_call, reply]) * 12000 / expected_input)
    assert usage.estimation_method == "calibrated"
    assert next(item.tokens for item in usage.categories if item.key == "assistant") == expected_assistant
    assert usage.used_tokens == sum(item.tokens for item in usage.categories)
    assert model.call_count == 2


async def test_loaded_skills_survive_source_removal_and_process_restart(config, tmp_path):
    skill = put_skill(config, "demo", "Loaded instructions " * 100)
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        response(call("load_skill", {"name": "demo"}, "one")),
        response(call("load_skill", {"name": "demo"}, "two")),
        AIMessage(content="Done", usage_metadata={"input_tokens": 5000, "output_tokens": 1, "total_tokens": 5001}),
    ])
    result = await ask(config, session.session_id, "Load twice", model=model)
    assert result.status == "succeeded", result.error
    skill.unlink()
    usage = await get_context_usage(config, session.session_id)
    assert next(item.tokens for item in usage.categories if item.key == "skills") > 500
    script = """
import asyncio, json, sys
from rs_harness.config import Config
from rs_harness.application import get_context_usage
usage = asyncio.run(get_context_usage(Config.load(sys.argv[1]), sys.argv[2]))
print(usage.model_dump_json())
"""
    process = await asyncio.to_thread(subprocess.run,
        [sys.executable, "-c", script, str(tmp_path / "config.toml"), session.session_id],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == usage.model_dump()


async def test_enabled_mcp_schemas_are_offline_and_match_model(config, tmp_path, monkeypatch):
    install_servers(monkeypatch, tmp_path)
    config = config.model_copy(update={"mcp": settings(docs={}, deploy={})})
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([response(call("connect_mcp", {"name": "docs"}, "connect")), AIMessage(content="Done")])
    result = await ask(config, session.session_id, "Connect docs", model=model)
    assert result.status == "succeeded", result.error
    before = (tmp_path / "docs.jsonl").read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("/context reconnected MCP")
    monkeypatch.setattr(MCPRegistry, "_make_adapter", forbidden)
    state = (await saved_state(config, session.session_id)).checkpoint["channel_values"]
    request = build_agent_components(config, workdir=str(tmp_path)).context_request(state)
    schemas = [convert_to_openai_tool(tool) for tool in request.tools]
    assert sorted(schemas, key=str) == sorted([convert_to_openai_tool(tool) for tool in model.tool_history[-1]], key=str)
    usage = await get_context_usage(config, session.session_id)
    assert next(item.tokens for item in usage.categories if item.key == "tool_schemas") == count_tokens_approximately([], tools=schemas)
    assert (tmp_path / "docs.jsonl").read_bytes() == before
    assert not (tmp_path / "deploy.jsonl").exists()


async def test_pending_root_writes_are_visible(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    await ask(config, session.session_id, "Question", model=ScriptedModel([AIMessage(content="Answer")]))
    saved = await saved_state(config, session.session_id)
    before = await get_context_usage(config, session.session_id)
    pending = HumanMessage(content="Pending message " * 100, id="pending")
    async with open_checkpointer(config) as saver:
        await saver.aput_writes(saved.config, [("messages", [pending])], "pending-task")
    after = await get_context_usage(config, session.session_id)
    assert after.used_tokens - before.used_tokens == count_tokens_approximately([pending])


async def test_tool_image_projection_is_counted_once_with_its_caption(config, tmp_path):
    from tests.images import write_test_image
    image = tmp_path / "sample.png"
    write_test_image(image)
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([response(call("read_image", {"path": "sample.png"}, "image")), AIMessage(content="Seen")])
    result = await ask(config, session.session_id, "Look", model=model)
    assert result.status == "succeeded", result.error
    usage = await get_context_usage(config, session.session_id)
    categories = {item.key: item.tokens for item in usage.categories}
    assert categories["images"] == 85
    assert categories["user"] == count_tokens_approximately([HumanMessage(content="Look")])
    assert categories["tool_results"] > 0


async def test_child_usage_and_private_history_do_not_change_root_meter(config, tmp_path, monkeypatch):
    from rs_harness import application
    from rs_harness.agents.main import build_agent

    child = ScriptedModel([AIMessage(content="Child result", usage_metadata={
        "input_tokens": 900000, "output_tokens": 1000, "total_tokens": 901000,
    })])
    root = ScriptedModel([
        response(call("delegate_task", {"agent_id": "worker", "query": "Do work"}, "delegate")),
        AIMessage(content="Root result", usage_metadata={"input_tokens": 8000, "output_tokens": 3, "total_tokens": 8003}),
    ])
    monkeypatch.setattr(application, "build_agent", lambda *args, **kwargs: build_agent(*args, **kwargs, subagent_model=child))
    session = create_session(config, workdir=str(tmp_path))
    result = await ask(config, session.session_id, "Delegate", model=root)
    assert result.status == "succeeded", result.error
    state = (await saved_state(config, session.session_id)).checkpoint["channel_values"]
    assert state["context_calibration"]["input_tokens"] == 8000
    assert not any(isinstance(item, AIMessage) and item.usage_metadata and item.usage_metadata["input_tokens"] == 900000
                   for item in state["messages"])
    usage = await get_context_usage(config, session.session_id)
    assert 8000 < usage.used_tokens < 9000
