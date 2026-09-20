import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rs_harness.agents.main import build_agent
from rs_harness.agents.middleware import StopHookMiddleware, build_middleware
from rs_harness.application import ask, create_session, get_pending_approvals, resume
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.models import ScriptedModel


@pytest.fixture
def install_stop_hooks(monkeypatch):
    def install(*hooks):
        def build_with_hooks(config):
            middleware = build_middleware(config)
            middleware[-1] = StopHookMiddleware(hooks)
            return middleware

        monkeypatch.setattr("rs_harness.agents.main.build_middleware", build_with_hooks)

    return install


async def checkpoint_messages(config, session):
    async with open_checkpointer(config) as saver:
        agent = build_agent(
            config, workdir=session.meta.workdir, checkpointer=saver,
            model=ScriptedModel([AIMessage(content="unused")]),
        )
        snapshot = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
    return snapshot.values["messages"]


@pytest.mark.parametrize("with_checks", [False, True])
async def test_empty_or_passing_checks_do_not_continue(config, install_stop_hooks, with_checks):
    checks = []

    def first(messages):
        checks.append(("first", messages[-1].content))
        return None

    async def second(messages):
        checks.append(("second", messages[-1].content))
        return None

    if with_checks:
        install_stop_hooks(first, second)
    session = create_session(config)
    model = ScriptedModel([AIMessage(content="done")])
    result = await ask(config, session.session_id, "run", model=model)

    assert result.status == "succeeded", result.error
    assert result.answer == "done"
    assert model.call_count == 1
    assert checks == ([("first", "done"), ("second", "done")] if with_checks else [])
    assert [(message.type, message.content) for message in await checkpoint_messages(config, session)] == [
        ("human", "run"), ("ai", "done"),
    ]


@pytest.mark.parametrize("feedback", ["", "continue", " "])
async def test_checks_are_ordered_and_first_non_none_short_circuits(config, install_stop_hooks, feedback):
    checks = []

    def first(messages):
        checks.append(("first", messages[-1].content))
        return None

    async def second(messages):
        checks.append(("second", messages[-1].content))
        return feedback if messages[-1].content == "draft" else None

    def third(messages):
        checks.append(("third", messages[-1].content))
        return None

    install_stop_hooks(first, second, third)
    session = create_session(config)
    model = ScriptedModel([AIMessage(content="draft"), AIMessage(content="done")])
    result = await ask(config, session.session_id, "run", model=model)

    assert result.status == "succeeded", result.error
    assert result.answer == ("done" if feedback else "draft")
    assert model.call_count == (2 if feedback else 1)
    expected_checks = [("first", "draft"), ("second", "draft")]
    expected_messages = [("human", "run"), ("ai", "draft")]
    if feedback:
        expected_checks += [("first", "done"), ("second", "done"), ("third", "done")]
        expected_messages.append(("human", feedback))
        assert [(message.type, message.content) for message in model.inputs[1][1:]] == expected_messages
        expected_messages.append(("ai", "done"))
    assert checks == expected_checks
    assert [(message.type, message.content) for message in await checkpoint_messages(config, session)] == expected_messages


async def test_continuation_can_run_tools_and_preserves_one_interaction(config, tmp_path, install_stop_hooks):
    checks = []

    def check(messages):
        checks.append(messages[-1].content)
        return "write the file" if messages[-1].content == "draft" else None

    install_stop_hooks(check)
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        AIMessage(content="draft"),
        AIMessage(content="writing", tool_calls=[{
            "name": "write_file", "args": {"path": "note.txt", "content": "written"},
            "id": "write", "type": "tool_call",
        }]),
        AIMessage(content="done"),
    ])
    result = await ask(config, session.session_id, "run", model=model)

    assert result.status == "succeeded", result.error
    assert result.answer == "done"
    assert checks == ["draft", "done"]
    assert model.call_count == 3
    assert (tmp_path / "note.txt").read_text() == "written"
    tool_message = next(message for message in model.inputs[-1] if isinstance(message, ToolMessage))
    assert tool_message.tool_call_id == "write"
    assert tool_message.status == "success"
    messages = await checkpoint_messages(config, session)
    assert [message.type for message in messages] == ["human", "ai", "human", "ai", "tool", "ai"]
    assert [message.content for message in messages if isinstance(message, HumanMessage)] == ["run", "write the file"]
    records = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert {record["interaction_id"] for record in records} == {result.interaction_id}
    assert sum(record["event"] == "user_input" for record in records) == 1
    assert sum(record["event"] == "interaction_succeeded" for record in records) == 1
    assert records[-1]["payload"]["answer"] == "done"


async def test_continuation_shares_model_call_budget(config, install_stop_hooks):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 2})})
    checks = []

    def check(messages):
        checks.append(messages[-1].content)
        return "continue"

    install_stop_hooks(check)
    session = create_session(config)
    model = ScriptedModel([AIMessage(content="first draft"), AIMessage(content="second draft")])
    result = await ask(config, session.session_id, "run", model=model)

    assert result.status == "failed"
    assert result.answer is None
    assert result.error["type"] == "ModelCallLimitExceededError"
    assert model.call_count == 2
    assert checks == ["first draft", "second draft"]
    records = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert records[-1]["event"] == "interaction_failed"
    assert not any(record["event"] == "interaction_succeeded" for record in records)


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_continuation_survives_approval_and_agent_rebuild(config, tmp_path, install_stop_hooks, decision):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 2})})
    checks = []

    async def check(messages):
        checks.append(messages[-1].content)
        return "write the file" if messages[-1].content == "draft" else None

    install_stop_hooks(check)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    session = create_session(config, workdir=str(workdir))
    model = ScriptedModel([
        AIMessage(content="draft"),
        AIMessage(content="writing", tool_calls=[{
            "name": "write_file", "args": {"path": "../external.txt", "content": "written"},
            "id": "write", "type": "tool_call",
        }]),
        AIMessage(content="done"),
    ])
    pending = await ask(config, session.session_id, "run", model=model)

    assert pending.status == "interrupted", pending.error
    assert pending.answer is None
    assert checks == ["draft"]
    assert model.call_count == 2
    assert not (tmp_path / "external.txt").exists()
    assert [message.content for message in await checkpoint_messages(config, session) if isinstance(message, HumanMessage)] == [
        "run", "write the file",
    ]
    queried = await get_pending_approvals(config, session.session_id, model=model)
    assert queried.interrupts == pending.interrupts
    assert checks == ["draft"]
    result = await resume(config, session.session_id, {
        queried.interrupts[0]["id"]: {"decisions": [{"type": decision}]},
    }, model=model)

    assert result.status == "succeeded", result.error
    assert result.answer == "done"
    assert result.interaction_id == pending.interaction_id
    assert model.call_count == 3
    assert checks == ["draft", "done"]
    assert (tmp_path / "external.txt").exists() == (decision == "approve")
    tool_message = next(message for message in model.inputs[-1] if isinstance(message, ToolMessage))
    assert tool_message.tool_call_id == "write"
    assert tool_message.status == ("success" if decision == "approve" else "error")
    assert [message.content for message in await checkpoint_messages(config, session) if isinstance(message, HumanMessage)] == [
        "run", "write the file",
    ]
    records = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert {record["interaction_id"] for record in records} == {pending.interaction_id}
    assert sum(record["event"] == "user_input" for record in records) == 1
    assert sum(record["event"] == "interaction_succeeded" for record in records) == 1


@pytest.mark.parametrize("invalid_result", [False, 0, ["continue"]])
async def test_invalid_hook_result_fails_interaction(config, install_stop_hooks, invalid_result):
    install_stop_hooks(lambda messages: invalid_result)
    session = create_session(config)
    model = ScriptedModel([AIMessage(content="done")])
    result = await ask(config, session.session_id, "run", model=model)

    assert result.status == "failed"
    assert result.answer is None
    assert result.error["type"] == "TypeError"
    assert "Stop hook" in result.error["message"]
    assert model.call_count == 1


@pytest.mark.parametrize("failure_at", ["model", "hook"])
@pytest.mark.parametrize("error", [RuntimeError("failure"), asyncio.CancelledError()])
async def test_errors_and_cancelled_nodes_are_not_success(config, install_stop_hooks, failure_at, error):
    checks = []

    async def check(messages):
        checks.append(messages[-1].content)
        raise error

    class FailingModel(ScriptedModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            raise error

    install_stop_hooks(check)
    session = create_session(config)
    model_class = FailingModel if failure_at == "model" else ScriptedModel
    model = model_class([AIMessage(content="done")])
    result = await ask(config, session.session_id, "run", model=model)
    assert result.status == "failed"
    assert result.answer is None
    if isinstance(error, RuntimeError):
        assert result.error == {"type": "RuntimeError", "message": "failure"}
    assert checks == (["done"] if failure_at == "hook" else [])
    records = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert not any(record["event"] == "interaction_succeeded" for record in records)


@pytest.mark.parametrize("cancel_at", ["model", "hook"])
async def test_task_cancellation_propagates(config, install_stop_hooks, cancel_at):
    entered = asyncio.Event()
    checks = []

    async def wait_for_cancellation():
        entered.set()
        await asyncio.Event().wait()

    async def check(messages):
        checks.append(messages[-1].content)
        await wait_for_cancellation()

    class WaitingModel(ScriptedModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            await wait_for_cancellation()

    install_stop_hooks(check)
    session = create_session(config)
    model_class = WaitingModel if cancel_at == "model" else ScriptedModel
    model = model_class([AIMessage(content="done")])
    interaction = asyncio.create_task(ask(config, session.session_id, "run", model=model))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        interaction.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interaction
    finally:
        interaction.cancel()
        await asyncio.gather(interaction, return_exceptions=True)
    assert checks == (["done"] if cancel_at == "hook" else [])
    records = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert not any(record["event"] == "interaction_succeeded" for record in records)


@pytest.mark.parametrize("messages", [
    [],
    [HumanMessage(content="run")],
    [AIMessage(content="working", tool_calls=[{"name": "glob", "args": {"pattern": "*"}, "id": "glob", "type": "tool_call"}])],
    [ToolMessage(content="done", tool_call_id="glob")],
])
async def test_stop_checks_ignore_non_final_states(messages):
    checks = []
    middleware = StopHookMiddleware([lambda history: checks.append(history)])

    assert await middleware.aafter_agent({"messages": messages}, None) is None
    assert checks == []
