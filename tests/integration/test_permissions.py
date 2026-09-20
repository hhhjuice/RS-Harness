import asyncio
import json
import subprocess
import sys

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from rs_harness import application
from rs_harness.agents.main import build_agent
from rs_harness.agents.middleware import build_middleware
from rs_harness.application import ask, create_session, get_pending_approvals, resume
from rs_harness.context import AgentContext
from rs_harness.inputs import build_human_message
from rs_harness.sessions.catalog import SessionBusy, SessionCatalog
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.images import write_test_image
from tests.models import ScriptedModel


def call(name, arguments, call_id):
    return {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}


def decide(pending, *decisions):
    assert len(pending.interrupts) == 1
    return {pending.interrupts[0]["id"]: {"decisions": list(decisions)}}


async def snapshot(config, session):
    async with open_checkpointer(config) as saver:
        agent = build_agent(config, workdir=session.meta.workdir, checkpointer=saver, model=ScriptedModel([AIMessage(content="unused")]))
        return await agent.aget_state({"configurable": {"thread_id": session.session_id}})


def events(session):
    return [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]


async def test_mixed_batch_only_reviews_ask_and_preserves_call_identity(config, tmp_path, monkeypatch):
    executed = []

    @tool("bash")
    async def recording_bash(command: str) -> str:
        """Record commands without executing a shell."""
        executed.append(command)
        return command

    monkeypatch.setattr("rs_harness.agents.main.bash", recording_bash)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    session = create_session(config, workdir=str(workdir))
    model = ScriptedModel([
        AIMessage(content="Not a final answer", tool_calls=[
            call("bash", {"command": "sudo rm forbidden"}, "deny"),
            call("write_file", {"path": "local.txt", "content": "local"}, "allow"),
            call("bash", {"command": "rm cleanup.txt"}, "approve"),
            call("bash", {"command": "chmod 777 file"}, "reject"),
            call("write_file", {"path": "../external.txt", "content": "outside"}, "external"),
        ]), AIMessage(content="done"),
    ])
    pending = await ask(config, session.session_id, "run", model=model)
    assert pending.status == "interrupted", pending.error
    assert pending.answer is pending.error is None
    actions = pending.interrupts[0]["action_requests"]
    assert [action["args"] for action in actions] == [
        {"command": "rm cleanup.txt"}, {"command": "chmod 777 file"},
        {"path": "../external.txt", "content": "outside"},
    ]
    assert not (workdir / "local.txt").exists()
    assert not (tmp_path / "external.txt").exists()
    assert executed == []
    assert not any(event["event"] in {"tool_start", "interaction_succeeded"} for event in events(session))
    result = await resume(config, session.session_id, decide(
        pending, {"type": "approve"}, {"type": "reject", "message": "keep permissions"}, {"type": "approve"},
    ), model=model)
    assert result.status == "succeeded", result.error
    assert result.interaction_id == pending.interaction_id
    assert executed == ["rm cleanup.txt"]
    assert (workdir / "local.txt").read_text() == "local"
    assert (tmp_path / "external.txt").read_text() == "outside"
    messages = {message.tool_call_id: message for message in model.inputs[-1] if isinstance(message, ToolMessage)}
    assert messages["deny"].status == messages["reject"].status == "error"
    assert "sudo" in messages["deny"].content
    assert "keep permissions" in messages["reject"].content
    assert json.loads(messages["external"].content)["path"] == str((tmp_path / "external.txt").resolve())
    assert json.loads(messages["allow"].content)["path"] == "local.txt"
    records = events(session)
    assert {event["interaction_id"] for event in records} == {pending.interaction_id}
    assert len([event for event in records if event["event"] == "user_input"]) == 1
    assert len([event for event in records if event["event"] == "tool_start"]) == 3
    assert len({event["run_id"] for event in records if event["event"] == "model_start"}) == 2


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("name", ["read_file", "write_file", "edit_file", "bash"])
async def test_real_dangerous_operation_requires_current_call_approval(config, tmp_path, name, decision):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    target = (workdir if name == "bash" else tmp_path) / "target.txt"
    target.write_text("original")
    arguments = {
        "read_file": {"path": "../target.txt"},
        "write_file": {"path": "../target.txt", "content": "changed"},
        "edit_file": {"path": "../target.txt", "old_text": "original", "new_text": "changed"},
        "bash": {"command": "rm target.txt"},
    }[name]
    model = ScriptedModel([AIMessage(content="", tool_calls=[call(name, arguments, "action")]), AIMessage(content="done")])
    session = create_session(config, workdir=str(workdir))
    pending = await ask(config, session.session_id, "run", model=model)
    assert pending.status == "interrupted", pending.error
    assert target.read_text() == "original"
    result = await resume(config, session.session_id, decide(pending, {"type": decision}), model=model)
    assert result.status == "succeeded", result.error
    message = next(message for message in model.inputs[-1] if isinstance(message, ToolMessage))
    assert message.status == ("success" if decision == "approve" else "error")
    if decision == "reject":
        assert target.read_text() == "original"
        assert not any(event["event"] == "tool_start" for event in events(session))
    elif name == "bash":
        assert not target.exists()
    else:
        assert json.loads(message.content)["path"] == str(target.resolve())
        assert target.read_text() == ("original" if name == "read_file" else "changed")
    duplicate = await resume(config, session.session_id, decide(pending, {"type": decision}), model=model)
    assert duplicate.error["type"] == "no_pending_approval"
    assert model.call_count == 2


@pytest.mark.parametrize("invalid", [
    {}, [], None, {"stale": {"decisions": [{"type": "approve"}]}},
    {"current": {"decisions": []}}, {"current": {"decisions": [{"type": "approve"}] * 2}},
    {"current": {"decisions": [{"type": "edit", "edited_action": {}}]}},
    {"current": {"decisions": [{"type": "respond"}]}},
    {"current": {"decisions": [{"type": "approve", "args": {}}]}},
    {"current": {"decisions": [{"type": "approve", "message": "extra"}]}},
    {"current": {"decisions": [{"type": "reject", "message": 4}]}},
    {"current": {"decisions": [{"type": "reject", "message": None}]}},
    {"current": {"decisions": [{"type": "approve"}], "extra": True}},
    {"current": {"decisions": [{"type": "approve"}]}, "extra": {"decisions": []}},
])
async def test_invalid_decisions_and_new_input_never_mutate_pending(config, tmp_path, invalid):
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("bash", {"command": "rm target.txt"}, "action")])])
    pending = await ask(config, session.session_id, "first", model=model)
    assert pending.status == "interrupted", pending.error
    earlier = await snapshot(config, session)
    trajectory = session.trajectory_path.read_bytes()
    metadata = session.state_path.read_bytes()
    if isinstance(invalid, dict):
        invalid = {pending.interrupts[0]["id"] if key == "current" else key: value for key, value in invalid.items()}
    failed = await resume(config, session.session_id, invalid, model=model)
    assert failed.error["type"] == "invalid_approval_decisions"
    assert failed.interaction_id == pending.interaction_id
    blocked = await ask(config, session.session_id, "new input", model=model)
    assert blocked.error["type"] == "session_pending_approval"
    assert blocked.interrupts == pending.interrupts
    assert (await get_pending_approvals(config, session.session_id, model=model)).interrupts == pending.interrupts
    assert await snapshot(config, session) == earlier
    assert session.trajectory_path.read_bytes() == trajectory
    assert session.state_path.read_bytes() == metadata
    assert model.call_count == 1


async def test_repeated_interrupts_budget_and_changed_cwd(config, tmp_path, monkeypatch):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 1})})
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    session = create_session(config, workdir=str(workdir))
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("write_file", {"path": "../one.txt", "content": "one"}, "one")]),
        AIMessage(content="", tool_calls=[call("write_file", {"path": "../two.txt", "content": "two", "approved": True, "allow_outside": True, "runtime": {"context": {"workdir": str(tmp_path)}}}, "two")]),
        AIMessage(content="done"),
    ])
    first = await ask(config, session.session_id, "run", model=model)
    monkeypatch.chdir(tmp_path.parent)
    second = await resume(config, session.session_id, decide(first, {"type": "approve"}), model=model)
    assert second.status == "interrupted", second.error
    assert (tmp_path / "one.txt").read_text() == "one"
    assert not (tmp_path / "two.txt").exists()
    assert first.interrupts[0]["id"] != second.interrupts[0]["id"]
    third = await resume(config, session.session_id, decide(second, {"type": "approve"}), model=model)
    assert third.status == "succeeded", third.error
    assert third.interaction_id == second.interaction_id == first.interaction_id
    assert (tmp_path / "two.txt").read_text() == "two"
    assert model.call_count == 3
    assert len([event for event in events(session) if event["event"] == "tool_start"]) == 2
    assert sum(isinstance(message, HumanMessage) for message in model.inputs[-1]) == 1


@pytest.mark.parametrize("condition", ["missing", "tampered", "malformed", "unwritable"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_approval_facts_do_not_depend_on_trajectory(config, tmp_path, condition, decision):
    session = create_session(config, workdir=str(tmp_path))
    target = tmp_path / "target.txt"
    target.write_text("keep")
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("bash", {"command": "rm target.txt"}, "action")]), AIMessage(content="done")])
    pending = await ask(config, session.session_id, "run", model=model)
    if condition in {"missing", "unwritable"}:
        session.trajectory_path.unlink()
        if condition == "unwritable":
            session.trajectory_path.mkdir()
    else:
        session.trajectory_path.write_text('{"interaction_id":"fake","event":"approved"}\n' if condition == "tampered" else "broken JSON\n")
    queried = await get_pending_approvals(config, session.session_id, model=model)
    assert queried.interaction_id == pending.interaction_id
    result = await resume(config, session.session_id, decide(queried, {"type": decision}), model=model)
    assert result.status == "succeeded", result.error
    assert result.interaction_id == pending.interaction_id
    assert target.exists() == (decision == "reject")
    assert bool(result.recording_errors) == (condition == "unwritable")
    assert model.call_count == 2


async def test_busy_closed_empty_and_isolated_sessions(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    empty = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("bash", {"command": "rm target.txt"}, "action")])])
    assert (await resume(config, empty.session_id, {}, model=model)).error["type"] == "no_pending_approval"
    pending = await ask(config, session.session_id, "run", model=model)
    decisions = decide(pending, {"type": "reject"})
    assert not (await get_pending_approvals(config, empty.session_id, model=model)).interrupts
    with session.interaction_lock():
        assert (await resume(config, session.session_id, decisions, model=model)).error["type"] == "session_busy"
        assert (await get_pending_approvals(config, session.session_id, model=model)).interrupts
        with pytest.raises(SessionBusy):
            SessionCatalog.close(config, session.session_id)
    SessionCatalog.close(config, session.session_id)
    assert (await resume(config, session.session_id, decisions, model=model)).error["type"] == "session_closed"
    assert (await ask(config, session.session_id, "new", model=model)).error["type"] == "session_closed"
    assert (await get_pending_approvals(config, session.session_id, model=model)).session_status == "closed"


async def test_image_history_survives_pending_and_source_deletion(config, tmp_path):
    image_path = tmp_path / "sample.png"
    write_test_image(image_path)
    message = build_human_message("inspect", images=[str(image_path)], config=config)
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("bash", {"command": "rm sample.png"}, "action")]), AIMessage(content="remembered")])
    pending = await ask(config, session.session_id, message, model=model)
    image_path.unlink()
    result = await resume(config, session.session_id, decide(pending, {"type": "reject"}), model=model)
    assert result.status == "succeeded", result.error
    humans = [message for message in model.inputs[-1] if isinstance(message, HumanMessage)]
    assert len(humans) == 1
    assert humans[0].content == message.content


@pytest.mark.parametrize("requires_approval", [False, True])
async def test_conditional_tools_keep_framework_parallel_execution(config, tmp_path, requires_approval):
    started = set()
    ready = asyncio.Event()

    @tool("bash")
    async def barrier(command: str) -> str:
        """Wait until both tool calls start; no shell is executed."""
        started.add(command)
        if len(started) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 2)
        return command

    prefix = "rm " if requires_approval else "echo "
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("bash", {"command": prefix + name}, name) for name in ("one", "two")]), AIMessage(content="done")])
    agent = create_agent(model, tools=[barrier], middleware=build_middleware(config), checkpointer=InMemorySaver(), context_schema=AgentContext)
    run_config = {"configurable": {"thread_id": "parallel"}}
    context = AgentContext(str(tmp_path))
    result = await agent.ainvoke({"messages": [HumanMessage("run")]}, run_config, context=context)
    if requires_approval:
        from langgraph.types import Command

        assert started == set()
        pending = result["__interrupt__"][0]
        result = await agent.ainvoke(Command(resume={pending.id: {"decisions": [{"type": "approve"}] * 2}}), run_config, context=context)
    assert result["messages"][-1].content == "done"
    assert started == {prefix + "one", prefix + "two"}


async def test_multiple_real_interrupt_ids_route_in_reverse_mapping_order(config, tmp_path, monkeypatch):
    class ProbeState(MessagesState):
        left: str
        right: str

    def branch(name):
        def review(state):
            decision = interrupt({
                "action_requests": [{"name": name, "args": {}}],
                "review_configs": [{"action_name": name, "allowed_decisions": ["approve", "reject"]}],
            })
            return {name: decision["decisions"][0]["type"]}
        return review

    graph = StateGraph(ProbeState)
    for name in ("left", "right"):
        graph.add_node(name, branch(name))
        graph.add_edge(START, name)
    graph.add_node("finish", lambda state: {"messages": [AIMessage(content=f'{state["left"]}/{state["right"]}')]})
    graph.add_edge(["left", "right"], "finish")
    graph.add_edge("finish", END)
    monkeypatch.setattr(application, "build_agent", lambda config, *, workdir, checkpointer, model: graph.compile(checkpointer=checkpointer))
    session = create_session(config, workdir=str(tmp_path))
    pending = await ask(config, session.session_id, "run")
    assert len(pending.interrupts) == 2
    decisions = {
        item["id"]: {"decisions": [{"type": "approve" if item["action_requests"][0]["name"] == "left" else "reject"}]}
        for item in reversed(pending.interrupts)
    }
    result = await resume(config, session.session_id, decisions)
    assert result.status == "succeeded", result.error
    assert result.answer == "approve/reject"
    assert result.interaction_id == pending.interaction_id


async def test_cli_checkpoint_survives_separate_processes(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    (tmp_path / "target.txt").write_text("delete once")
    script = '''
import sys
from langchain_core.messages import AIMessage
from rs_harness import cli, application
from tests.models import ScriptedModel

mode = sys.argv[1]
del sys.argv[1]
model = ScriptedModel([AIMessage(content="", tool_calls=[{"name":"bash", "args":{"command":"rm target.txt"}, "id":"remove", "type":"tool_call"}])]) if mode == "ask" else ScriptedModel([AIMessage(content="done")])
async def ask(config, session_id, text):
    return await application.ask(config, session_id, text, model=model)
async def resume(config, session_id, decisions):
    return await application.resume(config, session_id, decisions, model=model)
cli.ask = ask
cli.resume = resume
cli.main()
'''
    options = ["--config", str(tmp_path / "config.toml"), "--session-id", session.session_id, "--json"]
    def run_process(mode, arguments, code):
        process = subprocess.run([sys.executable, "-c", script, mode, *arguments, *options], capture_output=True, text=True, timeout=20)
        assert process.returncode == code, process.stderr
        assert process.stderr == ""
        return json.loads(process.stdout)
    first = run_process("ask", ["ask", "remove"], 2)
    queried = run_process("query", ["approvals"], 0)
    assert queried["interrupts"] == first["interrupts"]
    decisions_file = tmp_path / "decisions.json"
    decisions_file.write_text(json.dumps({queried["interrupts"][0]["id"]: {"decisions": [{"type": "approve"}]}}))
    result = run_process("resume", ["resume", "--decisions-file", str(decisions_file)], 0)
    assert result["interaction_id"] == first["interaction_id"]
    assert not (tmp_path / "target.txt").exists()
    state = await snapshot(config, session)
    assert sum(isinstance(message, HumanMessage) for message in state.values["messages"]) == 1
    assert run_process("repeat", ["resume", "--decisions-file", str(decisions_file)], 1)["error"]["type"] == "no_pending_approval"


async def test_all_hard_denials_run_without_an_approval_interrupt(config, tmp_path, monkeypatch):
    @tool("bash")
    async def forbidden_handler(command: str) -> str:
        """Fail if a forbidden command reaches tool execution."""
        pytest.fail(f"Forbidden handler ran: {command}")

    monkeypatch.setattr("rs_harness.agents.main.bash", forbidden_handler)
    commands = ["rm -rf /", "sudo rm file", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("bash", {"command": command}, str(index)) for index, command in enumerate(commands)]),
        AIMessage(content="denied"),
    ])
    session = create_session(config, workdir=str(tmp_path))
    result = await ask(config, session.session_id, "run", model=model)
    assert result.status == "succeeded", result.error
    assert not result.interrupts
    messages = [message for message in model.inputs[-1] if isinstance(message, ToolMessage)]
    assert len(messages) == len(commands)
    assert all(message.status == "error" for message in messages)
    assert not any(event["event"] in {"tool_start", "approval_submitted", "interaction_interrupted"} for event in events(session))


@pytest.mark.parametrize("kind", ["parent", "absolute", "symlink", "symlink_parent"])
async def test_new_external_file_is_approved_using_resolved_path(config, tmp_path, kind):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    target = tmp_path / "new.txt"
    if kind == "parent":
        path = "../new.txt"
    elif kind == "absolute":
        path = str(target)
    elif kind == "symlink":
        (workdir / "link").symlink_to(target)
        path = "link"
    else:
        (workdir / "link").symlink_to(tmp_path, target_is_directory=True)
        path = "link/new.txt"
    session = create_session(config, workdir=str(workdir))
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("write_file", {"path": path, "content": "written"}, "write")]), AIMessage(content="done")])
    pending = await ask(config, session.session_id, "run", model=model)
    assert pending.status == "interrupted", pending.error
    assert str(target.resolve()) in pending.interrupts[0]["action_requests"][0]["description"]
    assert not target.exists()
    result = await resume(config, session.session_id, decide(pending, {"type": "approve"}), model=model)
    assert result.status == "succeeded", result.error
    assert target.read_text() == "written"


@pytest.mark.parametrize("metadata_value", [None, "", 42])
async def test_invalid_checkpoint_identity_never_falls_back_to_trajectory(config, tmp_path, metadata_value):
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("bash", {"command": "rm target.txt"}, "action")])])
    pending = await ask(config, session.session_id, "run", model=model)
    async with open_checkpointer(config) as saver:
        run_config = {"configurable": {"thread_id": session.session_id, "checkpoint_ns": ""}}
        saved = await saver.aget_tuple(run_config)
        metadata = {**saved.metadata, "interaction_id": metadata_value}
        await saver.aput(run_config, saved.checkpoint, metadata, {})
    trajectory = session.trajectory_path.read_bytes()
    result = await resume(config, session.session_id, decide(pending, {"type": "approve"}), model=model)
    assert result.error["type"] == "invalid_checkpoint_metadata"
    assert result.interaction_id is None
    assert result.interrupts == pending.interrupts
    assert session.trajectory_path.read_bytes() == trajectory
    assert model.call_count == 1


async def test_resume_holds_session_lock_through_tool_execution(config, tmp_path, monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    @tool("bash")
    async def blocked_tool(command: str) -> str:
        """Hold a test tool call until the test releases it."""
        started.set()
        await asyncio.wait_for(release.wait(), 5)
        return command

    monkeypatch.setattr("rs_harness.agents.main.bash", blocked_tool)
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("bash", {"command": "rm target.txt"}, "action")]), AIMessage(content="done")])
    pending = await ask(config, session.session_id, "run", model=model)
    decisions = decide(pending, {"type": "approve"})
    running = asyncio.create_task(resume(config, session.session_id, decisions, model=model))
    try:
        await asyncio.wait_for(started.wait(), 3)
        assert (await ask(config, session.session_id, "new", model=model)).error["type"] == "session_busy"
        assert (await resume(config, session.session_id, decisions, model=model)).error["type"] == "session_busy"
        with pytest.raises(SessionBusy):
            SessionCatalog.close(config, session.session_id)
    finally:
        release.set()
        result = await running
    assert result.status == "succeeded", result.error


@pytest.mark.parametrize("control_flow", [GraphInterrupt(()), asyncio.CancelledError()])
async def test_application_does_not_convert_framework_control_flow(config, tmp_path, monkeypatch, control_flow):
    original_build = application.build_agent

    def build(config, **kwargs):
        agent = original_build(config, **kwargs)
        async def invoke(*args, **kwargs):
            raise control_flow
        agent.ainvoke = invoke
        return agent

    monkeypatch.setattr(application, "build_agent", build)
    session = create_session(config, workdir=str(tmp_path))
    with pytest.raises(type(control_flow)):
        await ask(config, session.session_id, "run", model=ScriptedModel([AIMessage(content="unused")]))
    assert not any(event["event"] == "interaction_failed" for event in events(session))


async def test_workspace_overwrite_remains_direct(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    (tmp_path / "note.txt").write_text("original")
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("write_file", {"path": "note.txt", "content": "overwritten"}, "write")]),
        AIMessage(content="done"),
    ])
    result = await ask(config, session.session_id, "overwrite", model=model)
    assert result.status == "succeeded", result.error
    assert not result.interrupts
    assert (tmp_path / "note.txt").read_text() == "overwritten"
    assert not any(event["event"] in {"approval_submitted", "interaction_interrupted"} for event in events(session))
