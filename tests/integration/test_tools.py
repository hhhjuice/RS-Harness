import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphInterrupt

from rs_harness.agents.middleware import ToolDenyMiddleware, build_middleware
from rs_harness.application import ask, create_session, resume
from tests.models import ScriptedModel


def tool_turn(name, arguments, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}])


async def test_all_five_production_tools_and_recording(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        tool_turn("write_file", {"path": "note.txt", "content": "first"}, "write"),
        tool_turn("read_file", {"path": "note.txt"}, "read"),
        tool_turn("edit_file", {"path": "note.txt", "old_text": "first", "new_text": "second"}, "edit"),
        tool_turn("glob", {"pattern": "*.txt"}, "glob"),
        tool_turn("bash", {"command": "cat note.txt"}, "bash"),
        AIMessage(content="done"),
    ])
    result = await ask(config, session.session_id, "run tools", model=model)
    assert result.status == "succeeded", result.error
    assert (tmp_path / "note.txt").read_text() == "second"
    assert {bound.name for bound in model.bound_tools} == {"read_file", "write_file", "edit_file", "glob", "bash", "read_image", "write_todos", "delegate_task", "load_skill"}
    for bound in model.bound_tools:
        assert not {"runtime", "workdir", "bash_timeout_s", "max_tool_output_bytes"} & bound.tool_call_schema.model_fields.keys()
    messages = {message.tool_call_id: message for message in model.inputs[-1] if isinstance(message, ToolMessage)}
    assert json.loads(messages["read"].content)["content"] == "first"
    assert json.loads(messages["bash"].content)["stdout"] == "second"
    events = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    starts = {event["run_id"]: event for event in events if event["event"] == "tool_start"}
    ends = {event["run_id"]: event for event in events if event["event"] == "tool_end"}
    assert len(starts) == len(ends) == 5
    assert starts.keys() == ends.keys()
    for run_id, event in starts.items():
        assert event["parent_run_id"] == ends[run_id]["parent_run_id"]
        assert "runtime" not in event["payload"]["inputs"]
        assert ends[run_id]["payload"]["output"]["status"] == "success"


async def test_parameter_errors_path_errors_and_runtime_cannot_be_forged(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        tool_turn("read_file", {"path": "note.txt", "offset": True}, "invalid"),
        tool_turn("read_file", {"path": "../outside"}, "outside"),
        tool_turn("write_file", {"path": "note.txt", "content": "fixed", "runtime": {"context": {"workdir": "/"}}, "workdir": "/"}, "fixed"),
        AIMessage(content="recovered"),
    ])
    result = await ask(config, session.session_id, "run", model=model)
    assert result.status == "interrupted", result.error
    result = await resume(config, session.session_id, {
        result.interrupts[0]["id"]: {"decisions": [{"type": "reject", "message": "outside"}]},
    }, model=model)
    assert result.status == "succeeded", result.error
    assert (tmp_path / "note.txt").read_text() == "fixed"
    messages = {message.tool_call_id: message for message in model.inputs[-1] if isinstance(message, ToolMessage)}
    assert messages["invalid"].status == messages["outside"].status == "error"
    assert "outside" in messages["outside"].content
    assert messages["fixed"].status == "success"
    events = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert len([event for event in events if event["event"] == "tool_error"]) == 1
    assert len([event for event in events if event["event"] == "tool_end"]) == 1


async def test_multiple_real_calls_use_session_context(config, tmp_path, monkeypatch):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    session = create_session(config, workdir=str(workdir))
    monkeypatch.chdir(tmp_path)
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "write_file", "args": {"path": f"{name}.txt", "content": name}, "id": name, "type": "tool_call"}
            for name in ["one", "two"]
        ]),
        AIMessage(content="done"),
    ])
    result = await ask(config, session.session_id, "run", model=model)
    assert result.status == "succeeded", result.error
    messages = [message for message in model.inputs[-1] if isinstance(message, ToolMessage)]
    assert {message.tool_call_id for message in messages} == {"one", "two"}
    assert all(message.status == "success" for message in messages)
    assert (workdir / "one.txt").read_text() == "one"
    assert (workdir / "two.txt").read_text() == "two"
    assert not (tmp_path / "one.txt").exists()


async def test_default_async_calls_can_overlap(config):
    started = set()
    both_started = asyncio.Event()

    @tool
    async def barrier(name: str) -> str:
        """Wait for both independent invocations."""
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), 2)
        return name

    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "barrier", "args": {"name": name}, "id": name, "type": "tool_call"}
            for name in ["one", "two"]
        ]), AIMessage(content="done"),
    ])
    agent = create_agent(model, tools=[barrier], middleware=build_middleware(config, tools=config.agent.tools))
    result = await agent.ainvoke({"messages": [HumanMessage("run")]})
    assert {message.tool_call_id: message.content for message in result["messages"] if isinstance(message, ToolMessage)} == {"one": "one", "two": "two"}


@pytest.mark.parametrize("error", [RuntimeError("bug"), asyncio.CancelledError(), GraphInterrupt(())])
async def test_unexpected_errors_and_control_flow_are_not_converted(config, error):
    async def handler(request):
        raise error

    for middleware in build_middleware(config, tools=config.agent.tools):
        if isinstance(middleware, (ToolErrorMiddleware, ToolDenyMiddleware)):
            with pytest.raises(type(error)):
                await middleware.awrap_tool_call(SimpleNamespace(tool=None, tool_call={"name": "probe", "id": "probe"}), handler)


async def test_recording_failure_does_not_repeat_side_effect(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    session.trajectory_path.unlink()
    session.trajectory_path.mkdir()
    model = ScriptedModel([
        tool_turn("bash", {"command": "printf x >> count.txt"}, "once"), AIMessage(content="done"),
    ])
    result = await ask(config, session.session_id, "run", model=model)
    assert result.status == "succeeded", result.error
    assert result.recording_errors
    assert (tmp_path / "count.txt").read_text() == "x"
    assert model.call_count == 2


async def test_budget_preserves_completed_side_effect(config, tmp_path):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 1})})
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        tool_turn("write_file", {"path": "note.txt", "content": "written"}, "write"), AIMessage(content="new turn"),
    ])
    failed = await ask(config, session.session_id, "run", model=model)
    assert failed.error["type"] == "ModelCallLimitExceededError"
    assert (tmp_path / "note.txt").read_text() == "written"
    assert model.call_count == 1
    assert (await ask(config, session.session_id, "next", model=model)).status == "succeeded"
