import json
import subprocess
import sys
from collections import Counter

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rs_harness.agents.main import build_agent
from rs_harness.agents.middleware import StopHookMiddleware, build_middleware
from rs_harness.application import ask, create_session, get_pending_approvals, resume
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.models import ScriptedModel


INITIAL_TODOS = [
    {"content": "Inspect the input", "status": "in_progress"},
    {"content": "Write the result", "status": "pending"},
]


def call(name, arguments, call_id):
    return {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}


def todo_response(todos, call_id="plan"):
    return AIMessage(content="", tool_calls=[call("write_todos", {"todos": todos}, call_id)])


async def snapshot(config, session):
    async with open_checkpointer(config) as saver:
        agent = build_agent(
            config, workdir=session.meta.workdir, checkpointer=saver,
            model=ScriptedModel([AIMessage(content="unused")]),
        )
        return await agent.aget_state({"configurable": {"thread_id": session.session_id}})


def events(session):
    return [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]


@pytest.fixture
async def planned_session(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    result = await ask(config, session.session_id, "Make a plan", model=ScriptedModel([
        todo_response(INITIAL_TODOS, "seed"), AIMessage(content="Plan ready"),
    ]))
    assert result.status == "succeeded", result.error
    assert (await snapshot(config, session)).values["todos"] == INITIAL_TODOS
    return session


def run_cli(config_path, session_id, responses, *arguments, expected_code=0):
    script = '''
import json
import sys
from langchain_core.messages import AIMessage
from rs_harness import application, cli
from tests.models import ScriptedModel

model = ScriptedModel([AIMessage.model_validate(item) for item in json.loads(sys.argv.pop(1))])
async def ask(config, session_id, text, **kwargs):
    return await application.ask(config, session_id, text, model=model, **kwargs)
cli.ask = ask
cli.main()
'''
    process = subprocess.run(
        [sys.executable, "-c", script, json.dumps([message.model_dump(mode="json") for message in responses]),
         *arguments, "--config", str(config_path), "--session-id", session_id, "--json"],
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == expected_code, process.stderr
    assert process.stderr == ""
    return json.loads(process.stdout)


async def test_production_binding_and_simple_answer_without_plan(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="Hello")])
    result = await ask(config, session.session_id, "Hello", model=model)

    assert result.status == "succeeded", result.error
    assert result.answer == "Hello"
    assert model.call_count == 1
    assert Counter(tool.name for tool in model.bound_tools) == Counter([
        "write_todos", "read_file", "write_file", "edit_file", "glob", "bash", "read_image", "delegate_task", "load_skill",
    ])
    system = [message for message in model.inputs[0] if isinstance(message, SystemMessage)]
    assert len(system) == 1
    assert "You are a test assistant" in str(system[0].content)
    assert "write_todos" in str(system[0].content)
    assert "todos" not in (await snapshot(config, session)).values


async def test_plan_replacement_completion_and_clear(config, planned_session):
    updates = [
        [{"content": "Inspect the input", "status": "completed"},
         {"content": "Verify the output", "status": "in_progress"}],
        [{"content": "Verify the output", "status": "completed"}],
        [],
    ]
    for index, todos in enumerate(updates):
        call_id = f"update-{index}"
        model = ScriptedModel([todo_response(todos, call_id), AIMessage(content="Updated")])
        result = await ask(config, planned_session.session_id, "Update the plan", model=model)
        assert result.status == "succeeded", result.error
        state = await snapshot(config, planned_session)
        assert state.values["todos"] == todos
        messages = [message for message in state.values["messages"] if isinstance(message, ToolMessage) and message.tool_call_id == call_id]
        assert len(messages) == 1
        assert messages[0].status == "success"
        assert model.call_count == 2


@pytest.mark.parametrize("arguments", [
    {},
    {"todos": None},
    {"todos": "[]"},
    {"todos": {}},
    {"todos": [{"status": "pending"}]},
    {"todos": [{"content": "missing status"}]},
    {"todos": [{"content": "invalid status", "status": "done"}]},
    {"todos": [None]},
    {"todos": [{"content": 42, "status": "pending"}]},
    {"todos": [{"content": "valid", "status": "completed"}, {"content": "invalid", "status": "blocked"}]},
])
async def test_invalid_update_preserves_plan_and_can_be_corrected(config, planned_session, arguments):
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("write_todos", arguments, "invalid")]),
        AIMessage(content="Invalid update acknowledged"),
    ])
    result = await ask(config, planned_session.session_id, "Try an update", model=model)

    assert result.status == "succeeded", result.error
    state = await snapshot(config, planned_session)
    assert state.values["todos"] == INITIAL_TODOS
    errors = [message for message in state.values["messages"] if isinstance(message, ToolMessage) and message.tool_call_id == "invalid"]
    assert len(errors) == 1
    assert errors[0].status == "error"
    assert model.call_count == 2
    corrected = [{"content": "Verified", "status": "completed"}]
    result = await ask(config, planned_session.session_id, "Correct it", model=ScriptedModel([
        todo_response(corrected, "corrected"), AIMessage(content="Corrected"),
    ]))
    assert result.status == "succeeded", result.error
    assert (await snapshot(config, planned_session)).values["todos"] == corrected


@pytest.mark.parametrize("todos", [
    [{"content": "", "status": "pending"}],
    [{"content": "First", "status": "in_progress"}, {"content": "Second", "status": "in_progress"}],
    [{"content": f"Task {index}", "status": "pending"} for index in range(21)],
])
async def test_official_schema_does_not_add_tutorial_limits(config, planned_session, todos):
    result = await ask(config, planned_session.session_id, "Replace the plan", model=ScriptedModel([
        todo_response(todos), AIMessage(content="Updated"),
    ]))
    assert result.status == "succeeded", result.error
    assert (await snapshot(config, planned_session)).values["todos"] == todos


async def test_followup_preserves_plan_and_sessions_are_isolated(config, tmp_path, planned_session):
    model = ScriptedModel([AIMessage(content="Still planned")])
    result = await ask(config, planned_session.session_id, "What is next?", model=model)
    assert result.status == "succeeded", result.error
    assert (await snapshot(config, planned_session)).values["todos"] == INITIAL_TODOS
    assert any(isinstance(message, ToolMessage) and message.tool_call_id == "seed" for message in model.inputs[0])
    assert sum(isinstance(message, HumanMessage) for message in model.inputs[0]) == 2
    other = create_session(config, workdir=str(tmp_path))
    assert "todos" not in (await snapshot(config, other)).values
    result = await ask(config, other.session_id, "Hello", model=ScriptedModel([AIMessage(content="Hello")]))
    assert result.status == "succeeded", result.error
    assert "todos" not in (await snapshot(config, other)).values


async def test_planning_does_not_execute_content_or_require_approval(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    target = tmp_path / "keep.txt"
    target.write_text("keep")
    todos = [
        {"content": "bash: rm keep.txt", "status": "completed"},
        {"content": "write_file: ../outside.txt", "status": "pending"},
        {"content": "sudo shutdown", "status": "in_progress"},
    ]
    result = await ask(config, session.session_id, "Plan only", model=ScriptedModel([
        todo_response(todos), AIMessage(content="Plan only"),
    ]))

    assert result.status == "succeeded", result.error
    assert result.interrupts == []
    assert target.read_text() == "keep"
    assert not (tmp_path.parent / "outside.txt").exists()
    assert (await snapshot(config, session)).values["todos"] == todos
    records = events(session)
    starts = [record for record in records if record["event"] == "tool_start"]
    ends = [record for record in records if record["event"] == "tool_end"]
    assert len(starts) == len(ends) == 1
    assert starts[0]["payload"]["name"] == "write_todos"
    assert starts[0]["payload"]["inputs"]["todos"] == todos
    assert starts[0]["run_id"] == ends[0]["run_id"]
    assert starts[0]["parent_run_id"] == ends[0]["parent_run_id"]
    assert starts[0]["parent_run_id"] is not None
    assert "todos" in str(ends[0]["payload"]["output"])
    assert {record["interaction_id"] for record in records} == {result.interaction_id}
    assert not any(record["event"] == "interaction_interrupted" for record in records)


@pytest.mark.parametrize("duplicate", [False, True])
async def test_todo_updates_leave_independent_tools_runnable(config, tmp_path, planned_session, duplicate):
    todos = [{"content": "Independent step", "status": "in_progress"}]
    calls = [call("write_todos", {"todos": todos}, "first")]
    if duplicate:
        calls.append(call("write_todos", {"todos": []}, "second"))
    calls.append(call("write_file", {"path": "ordinary.txt", "content": "written"}, "write"))
    model = ScriptedModel([AIMessage(content="", tool_calls=calls), AIMessage(content="Done")])
    result = await ask(config, planned_session.session_id, "Run independent operations", model=model)

    assert result.status == "succeeded", result.error
    assert (tmp_path / "ordinary.txt").read_text() == "written"
    state = await snapshot(config, planned_session)
    assert state.values["todos"] == (INITIAL_TODOS if duplicate else todos)
    messages = [message for message in state.values["messages"] if isinstance(message, ToolMessage)]
    assert Counter(message.tool_call_id for message in messages) == Counter(["seed", *[item["id"] for item in calls]])
    assert next(message for message in messages if message.tool_call_id == "first").status == ("error" if duplicate else "success")
    starts = [record for record in events(planned_session) if record["event"] == "tool_start" and record["interaction_id"] == result.interaction_id]
    assert Counter(record["payload"]["name"] for record in starts) == Counter(["write_file"] if duplicate else ["write_todos", "write_file"])


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_todos_and_mixed_approval_survive_rebuild(config, tmp_path, planned_session, duplicate, decision):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 1})})
    target = tmp_path / "target.txt"
    protected = tmp_path / "protected.txt"
    target.write_text("remove if approved")
    protected.write_text("keep")
    todos = [{"content": "Review cleanup", "status": "in_progress"}]
    calls = [call("write_todos", {"todos": todos}, "first")]
    if duplicate:
        calls.append(call("write_todos", {"todos": []}, "second"))
    calls.extend([
        call("write_file", {"path": "ordinary.txt", "content": "written"}, "ordinary"),
        call("bash", {"command": "rm target.txt"}, "dangerous"),
        call("bash", {"command": "sudo rm protected.txt"}, "forbidden"),
    ])
    model = ScriptedModel([AIMessage(content="", tool_calls=calls)])
    pending = await ask(config, planned_session.session_id, "Review and clean", model=model)

    assert pending.status == "interrupted", pending.error
    assert model.call_count == 1
    assert target.exists() and protected.exists()
    assert not (tmp_path / "ordinary.txt").exists()
    before = await snapshot(config, planned_session)
    assert before.values["todos"] == INITIAL_TODOS
    assert len(pending.interrupts) == 1
    actions = pending.interrupts[0]["action_requests"]
    assert [(item["name"], item["args"]) for item in actions] == [("bash", {"command": "rm target.txt"})]
    assert not any(record["event"] == "tool_start" and record["interaction_id"] == pending.interaction_id for record in events(planned_session))
    records = planned_session.trajectory_path.read_bytes()
    gated = await ask(config, planned_session.session_id, "Replace the plan", model=model)
    assert gated.error["type"] == "session_pending_approval"
    assert planned_session.trajectory_path.read_bytes() == records
    assert (await snapshot(config, planned_session)).config == before.config
    queried = await get_pending_approvals(config, planned_session.session_id, model=model)
    assert queried.interrupts == pending.interrupts

    resumed_model = ScriptedModel([AIMessage(content="Reviewed")])
    result = await resume(config, planned_session.session_id, {
        queried.interrupts[0]["id"]: {"decisions": [{"type": decision}]},
    }, model=resumed_model)
    assert result.status == "succeeded", result.error
    assert result.interaction_id == pending.interaction_id
    assert resumed_model.call_count == 1
    assert target.exists() == (decision == "reject")
    assert protected.read_text() == "keep"
    assert (tmp_path / "ordinary.txt").read_text() == "written"
    state = await snapshot(config, planned_session)
    assert state.values["todos"] == (INITIAL_TODOS if duplicate else todos)
    messages = [message for message in state.values["messages"] if isinstance(message, ToolMessage)]
    assert Counter(message.tool_call_id for message in messages) == Counter(["seed", *[item["id"] for item in calls]])
    statuses = {message.tool_call_id: message.status for message in messages}
    assert statuses["first"] == ("error" if duplicate else "success")
    if duplicate:
        assert statuses["second"] == "error"
    assert statuses["dangerous"] == ("success" if decision == "approve" else "error")
    assert statuses["forbidden"] == "error"
    assert sum(isinstance(message, HumanMessage) for message in state.values["messages"]) == 2


async def test_budget_failure_keeps_written_plan_and_next_run_resets_budget(config, planned_session):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 1})})
    todos = [{"content": "Saved before the limit", "status": "in_progress"}]
    model = ScriptedModel([todo_response(todos), AIMessage(content="Unreachable")])
    result = await ask(config, planned_session.session_id, "Update", model=model)

    assert result.status == "failed"
    assert result.error["type"] == "ModelCallLimitExceededError"
    assert model.call_count == 1
    assert (await snapshot(config, planned_session)).values["todos"] == todos
    next_model = ScriptedModel([AIMessage(content="Continue later")])
    result = await ask(config, planned_session.session_id, "What is next?", model=next_model)
    assert result.status == "succeeded", result.error
    assert next_model.call_count == 1
    assert (await snapshot(config, planned_session)).values["todos"] == todos


@pytest.mark.parametrize("budget", [2, 3])
async def test_stop_hook_can_update_plan_with_same_interaction_and_budget(config, planned_session, monkeypatch, budget):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": budget})})
    checks = []

    def check(messages):
        checks.append(messages[-1].content)
        return "Update the plan" if messages[-1].content == "Draft" else None

    def with_hook(config, **kwargs):
        middleware = build_middleware(config, **kwargs)
        middleware[-1] = StopHookMiddleware((check,))
        return middleware

    monkeypatch.setattr("rs_harness.agents.main.build_middleware", with_hook)
    todos = [{"content": "Verified", "status": "completed"}]
    model = ScriptedModel([AIMessage(content="Draft"), todo_response(todos), AIMessage(content="Done")])
    result = await ask(config, planned_session.session_id, "Finish", model=model)

    assert model.call_count == budget
    assert (await snapshot(config, planned_session)).values["todos"] == todos
    assert result.status == ("succeeded" if budget == 3 else "failed")
    if budget == 2:
        assert result.error["type"] == "ModelCallLimitExceededError"
    assert checks == (["Draft", "Done"] if budget == 3 else ["Draft"])
    records = [record for record in events(planned_session) if record["interaction_id"] == result.interaction_id]
    assert sum(record["event"] == "user_input" for record in records) == 1
    assert sum(record["event"] == "tool_start" for record in records) == 1
    assert [message.content for message in model.inputs[1] if isinstance(message, HumanMessage)][-2:] == ["Finish", "Update the plan"]


@pytest.mark.parametrize("condition", ["missing", "malformed", "unwritable"])
async def test_plan_recovery_does_not_depend_on_trajectory(config, planned_session, condition):
    path = planned_session.trajectory_path
    if condition == "malformed":
        path.write_text("broken JSON\n")
    else:
        path.unlink()
        if condition == "unwritable":
            path.mkdir()
    assert (await snapshot(config, planned_session)).values["todos"] == INITIAL_TODOS
    todos = [{"content": "Recovered", "status": "completed"}]
    model = ScriptedModel([todo_response(todos), AIMessage(content="Recovered")])
    result = await ask(config, planned_session.session_id, "Finish", model=model)

    assert result.status == "succeeded", result.error
    assert bool(result.recording_errors) == (condition == "unwritable")
    assert model.call_count == 2
    assert (await snapshot(config, planned_session)).values["todos"] == todos


async def test_cli_todos_survive_independent_processes(config, tmp_path):
    session = create_session(config, workdir=str(tmp_path))
    first = run_cli(tmp_path / "config.toml", session.session_id, [
        todo_response(INITIAL_TODOS, "seed"), AIMessage(content="Planned"),
    ], "ask", "Plan")
    assert first["status"] == "succeeded"
    assert "todos" not in first
    assert (await snapshot(config, session)).values["todos"] == INITIAL_TODOS
    todos = [{"content": "Verified", "status": "completed"}]
    second = run_cli(tmp_path / "config.toml", session.session_id, [
        todo_response(todos), AIMessage(content="Done"),
    ], "ask", "Continue")
    assert second["status"] == "succeeded"
    assert second["interaction_id"] != first["interaction_id"]
    state = await snapshot(config, session)
    assert state.values["todos"] == todos
    assert [message.content for message in state.values["messages"] if isinstance(message, HumanMessage)] == ["Plan", "Continue"]
    assert Counter(message.tool_call_id for message in state.values["messages"] if isinstance(message, ToolMessage)) == Counter(["seed", "plan"])
