import asyncio
import json
import subprocess
import sys

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from rs_harness import application
from rs_harness.config import Config
from rs_harness.progress import CliProgress
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.integration.test_subagents import call, delegate, install_worker, records, request, snapshot, tool_messages
from tests.models import ScriptedModel


def configured(config, **updates):
    return config.model_copy(update={"subagents": (config.subagents[0].model_copy(update=updates),)})


def human_text(messages):
    return [message.content for message in messages if isinstance(message, HumanMessage)]


def assert_paired(messages):
    pending = set()
    for message in messages:
        if isinstance(message, AIMessage):
            assert not pending
            pending.update(item["id"] for item in message.tool_calls)
        elif isinstance(message, ToolMessage):
            assert message.tool_call_id in pending
            pending.remove(message.tool_call_id)
        else:
            assert not pending
    assert not pending


@pytest.mark.parametrize("tools", [(), ("read_file",), ("read_file", "write_todos")])
async def test_actual_tool_scope_and_independent_prompt(config, tmp_path, monkeypatch, tools):
    config = configured(config, tools=tools, name="资料核查员")
    config.subagents[0].system_prompt_file.write_text("Independent role {workdir}")
    worker = ScriptedModel([AIMessage(content="result")])
    parent = ScriptedModel([request(delegate()), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    assert {tool.name for tool in worker.bound_tools} == set(tools)
    assert "Independent role " + str(tmp_path) in str(worker.inputs[0][0].content)
    assert "资料核查员" in str(worker.inputs[0][0].content)
    assert "You are a test assistant" not in str(worker.inputs[0])
    description = next(tool.description for tool in parent.bound_tools if tool.name == "delegate_task")
    assert "资料核查员" in description and "worker" in description


async def test_unregistered_tools_cannot_execute_or_inject_todos(config, tmp_path, monkeypatch):
    config = configured(config, tools=("read_file",))
    worker = ScriptedModel([
        request(call("write_file", {"path": "forbidden", "content": "bad"}, "write"),
                call("write_todos", {"todos": [{"content": "forbidden", "status": "completed"}]}, "todo"),
                delegate("recursive", "recursive")),
        AIMessage(content="no access"),
    ])
    parent = ScriptedModel([request(call("delegate_task", {
        "agent_id": "worker", "query": "work", "tools": ["write_file"],
        "system_prompt": "evil", "workdir": "/other",
    }, "task")), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    assert worker.call_count == 2
    assert [m.status for m in tool_messages(worker.inputs[-1])] == ["error"] * 3
    assert not (tmp_path / "forbidden").exists()
    assert "evil" not in str(worker.inputs[0][0]) and "/other" not in str(worker.inputs[0][0])
    assert {tool.name for tool in worker.bound_tools} == {"read_file"}
    assert "todos" not in (await snapshot(config, session)).values


async def test_unknown_id_returns_correctable_error(config, tmp_path, monkeypatch):
    worker = ScriptedModel([AIMessage(content="result")])
    parent = ScriptedModel([request(delegate(agent_id="unknown")), request(delegate(call_id="valid")), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    error = tool_messages(parent.inputs[1])[0]
    assert error.status == "error" and "unknown_subagent" in error.content
    assert worker.call_count == 1


async def test_persistent_memory_and_todos_across_rounds_and_factory_rebuild(config, tmp_path, monkeypatch):
    config = configured(config, persistence="per_thread")
    (tmp_path / "evidence").write_text("private-detail-7319")
    todos = [{"content": "private-plan", "status": "in_progress"}]
    worker = ScriptedModel([
        request(call("read_file", {"path": "evidence"}, "read"), call("write_todos", {"todos": todos}, "plan")),
        AIMessage(content="checked"), AIMessage(content="followed up"), AIMessage(content="remembered"),
    ])
    parent = ScriptedModel([
        request(delegate("inspect evidence", "first")), request(delegate("more detail", "second")), AIMessage(content="one"),
        request(delegate("recall prior evidence", "third")), AIMessage(content="two"),
    ])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    for text in ("start", "follow up"):
        result = await application.ask(config, session.session_id, text, model=parent)
        assert result.status == "succeeded", result.error
    assert human_text(worker.inputs[-1]) == ["inspect evidence", "more detail", "recall prior evidence"]
    assert "private-detail-7319" in str(worker.inputs[-1])
    assert "private-detail-7319" not in str(parent.inputs)
    assert_paired(worker.inputs[-1])
    async with open_checkpointer(config) as saver:
        checkpoints = [item async for item in saver.alist({"configurable": {"thread_id": session.session_id}})]
    child_states = [item.checkpoint["channel_values"] for item in checkpoints
                    if "subagent_worker" in item.config["configurable"]["checkpoint_ns"]]
    assert any(state.get("todos") == todos and len(human_text(state.get("messages", []))) == 3 for state in child_states)
    assert not (await snapshot(config, session)).values.get("todos")


async def test_persistent_id_and_parent_session_isolation_with_reversed_order(config, tmp_path, monkeypatch):
    definitions = tuple(config.subagents[0].model_copy(update={"id": name, "name": name, "persistence": "per_thread"})
                        for name in ("alpha", "beta"))
    config = config.model_copy(update={"subagents": definitions})
    inputs = []

    class IdentityWorker(ScriptedModel):
        async def _agenerate(self, messages, **kwargs):
            inputs.append(list(messages))
            name = "alpha" if "Identity: alpha" in str(messages[0].content) else "beta"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=name + "-private-observation"))])

    install_worker(monkeypatch, IdentityWorker([AIMessage(content="unused")]))
    parent = ScriptedModel([
        request(delegate("first", "a1", "alpha"), delegate("first", "b1", "beta")), AIMessage(content="one"),
        request(delegate("next", "b2", "beta"), delegate("next", "a2", "alpha")), AIMessage(content="two"),
    ])
    first = application.create_session(config, workdir=str(tmp_path))
    for text in ("start", "continue"):
        result = await application.ask(config, first.session_id, text, model=parent)
        assert result.status == "succeeded", result.error
    for messages in inputs[2:]:
        identity = "alpha" if "Identity: alpha" in str(messages[0].content) else "beta"
        assert human_text(messages) == ["first", "next"]
        assert [m.content for m in messages if isinstance(m, AIMessage)] == [identity + "-private-observation"]
    second = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, second.session_id, "new session", model=parent)
    assert result.status == "succeeded", result.error
    assert all(human_text(messages) == ["first"] and len(messages) == 2 for messages in inputs[4:])


async def test_same_batch_duplicates_skip_execution_but_next_round_can_follow_up(config, tmp_path, monkeypatch):
    config = configured(config, persistence="per_thread")
    worker = ScriptedModel([AIMessage(content="first result"), AIMessage(content="follow-up result")])
    parent = ScriptedModel([
        request(delegate("first", "first"), delegate("must not run", "duplicate")),
        request(delegate("follow up", "followup")), AIMessage(content="done"),
    ])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    messages = tool_messages(parent.inputs[1])
    assert [(m.tool_call_id, m.status) for m in messages] == [("first", "success"), ("duplicate", "error")]
    assert "duplicate_persistent_subagent" in messages[1].content
    assert worker.call_count == 2 and human_text(worker.inputs[1]) == ["first", "follow up"]


def provider_error(kind):
    request = httpx.Request("POST", "https://example.invalid/model")
    if kind == "timeout":
        return APITimeoutError(request=request)
    if kind == "connection":
        return APIConnectionError(message="private-provider-details", request=request)
    if kind == "rate_limit":
        return RateLimitError("private-provider-details", response=httpx.Response(429, request=request), body=None)
    if kind == "auth":
        return AuthenticationError("auth failed", response=httpx.Response(401, request=request), body=None)
    if kind == "checkpoint":
        return OSError("checkpoint storage failed")
    if kind == "ordinary_timeout":
        return TimeoutError("unrelated timeout")
    return RuntimeError("unexpected bug")


@pytest.mark.parametrize("kind", ["timeout", "connection", "rate_limit", "auth", "checkpoint", "ordinary_timeout", "unknown"])
async def test_delegation_error_whitelist(config, tmp_path, monkeypatch, kind):
    attempts = []
    error = provider_error(kind)

    class FailingWorker(ScriptedModel):
        async def _agenerate(self, messages, **kwargs):
            attempts.append(messages)
            raise error

    worker = FailingWorker([AIMessage(content="unused")])
    install_worker(monkeypatch, worker)
    parent = ScriptedModel([request(delegate()), AIMessage(content="parent decides what to do")])
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert len(attempts) == 1
    if kind in {"timeout", "connection", "rate_limit"}:
        assert result.status == "succeeded", result.error
        message = tool_messages(parent.inputs[-1])[0]
        assert message.status == "error" and message.tool_call_id == "delegate"
        assert "worker" in message.content and "通用任务助手" in message.content
        assert "may already have changed files" in message.content
        assert "private-provider-details" not in message.content
    else:
        assert result.status == "failed" and result.error["type"] == type(error).__name__
        assert parent.call_count == 1


@pytest.mark.parametrize("failure", ["budget", "timeout"])
async def test_failed_persistent_child_can_follow_up_without_replaying_effects(config, tmp_path, monkeypatch, failure):
    config = configured(config, persistence="per_thread")
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 2})})
    attempts = []

    class RecoveringWorker(ScriptedModel):
        async def _agenerate(self, messages, **kwargs):
            attempts.append(list(messages))
            if failure == "timeout" and len(attempts) == 2:
                raise provider_error("timeout")
            return await super()._agenerate(messages, **kwargs)

    responses = [request(call("bash", {"command": "printf x >> marker"}, "effect"))]
    if failure == "budget":
        responses.append(request(call("read_file", {"path": "marker"}, "verify")))
    responses.append(AIMessage(content="recovered"))
    worker = RecoveringWorker(responses)
    install_worker(monkeypatch, worker)
    parent = ScriptedModel([
        request(delegate("perform bounded work", "original")), AIMessage(content="failed task reviewed"),
        request(delegate("inspect previous observations and report", "followup")), AIMessage(content="completed"),
    ])
    session = application.create_session(config, workdir=str(tmp_path))
    first = await application.ask(config, session.session_id, "start", model=parent)
    assert first.status == "succeeded", first.error
    message = tool_messages(parent.inputs[-1])[0]
    assert message.status == "error" and ("budget_exhausted" if failure == "budget" else "model_timeout") in message.content
    assert (tmp_path / "marker").read_text() == "x"
    final = await application.ask(config, session.session_id, "continue", model=parent)
    assert final.status == "succeeded", final.error
    assert (tmp_path / "marker").read_text() == "x"
    assert human_text(attempts[-1]) == ["perform bounded work", "inspect previous observations and report"]
    assert_paired(attempts[-1])
    assert tool_messages(parent.inputs[-1])[-1].content == "recovered"
    assert sum(row["event"] == "tool_start" and row["payload"]["name"] == "bash" for row in records(session)) == 1


@pytest.mark.parametrize("persistent", [False, True])
async def test_independent_children_overlap_and_recoverable_error_preserves_sibling(config, tmp_path, monkeypatch, persistent):
    if persistent:
        config = config.model_copy(update={"subagents": tuple(config.subagents[0].model_copy(update={
            "id": name, "name": name, "persistence": "per_thread",
        }) for name in ("alpha", "beta"))})
    entered = []
    ready = asyncio.Event()

    class ParallelWorker(ScriptedModel):
        async def _agenerate(self, messages, **kwargs):
            query = human_text(messages)[0]
            entered.append(query)
            if len(entered) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 5)
            if query == "fail":
                raise provider_error("connection")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="sibling result"))])

    install_worker(monkeypatch, ParallelWorker([AIMessage(content="unused")]))
    ids = ("alpha", "beta") if persistent else ("worker", "worker")
    parent = ScriptedModel([
        request(delegate("fail", "fail", ids[0]), delegate("succeed", "success", ids[1]),
                call("write_file", {"path": "parent-artifact", "content": "kept"}, "parent-tool")),
        AIMessage(content="done"),
    ])
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    messages = {m.tool_call_id: m for m in tool_messages(parent.inputs[1])}
    assert messages["fail"].status == "error" and messages["success"].content == "sibling result"
    assert messages["parent-tool"].status == "success" and (tmp_path / "parent-artifact").read_text() == "kept"


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("change", ["prompt", "id", "tools", "persistence"])
async def test_changed_execution_definitions_block_ask_and_resume(config, tmp_path, monkeypatch, pending, change):
    config = configured(config, persistence="per_thread")
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    worker = ScriptedModel([AIMessage(content="child result")])
    parent = ScriptedModel([
        request(call("write_file", {"path": "../external", "content": "must not be written"}, "external"))
        if pending else request(delegate()), AIMessage(content="done"),
    ])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(workdir))
    first = await application.ask(config, session.session_id, "start", model=parent)
    assert first.status == ("interrupted" if pending else "succeeded"), first.error
    counts = (parent.call_count, worker.call_count)
    trajectory = session.trajectory_path.read_bytes()
    if change == "prompt":
        config.subagents[0].system_prompt_file.write_text("new role")
        changed = config
    else:
        changed = configured(config, **{"id": {"id": "changed"}, "tools": {"tools": ()},
                                         "persistence": {"persistence": "per_invocation"}}[change])
    if pending:
        result = await application.resume(changed, session.session_id, {
            first.interrupts[0]["id"]: {"decisions": [{"type": "approve"}]},
        }, model=parent)
    else:
        result = await application.ask(changed, session.session_id, "continue", model=parent)
    assert result.error["type"] == "subagent_config_mismatch"
    assert (parent.call_count, worker.call_count) == counts
    assert session.trajectory_path.read_bytes() == trajectory
    assert not (tmp_path / "external").exists()


async def test_display_rename_preserves_identity_and_records_trusted_name(config, tmp_path, monkeypatch, capsys):
    config = configured(config, persistence="per_thread")
    worker = ScriptedModel([AIMessage(content="first result"), AIMessage(content="second result")])
    install_worker(monkeypatch, worker)
    parent = ScriptedModel([request(delegate("first", "first")), AIMessage(content="one"),
                            request(delegate("follow up", "second")), AIMessage(content="two")])
    session = application.create_session(config, workdir=str(tmp_path))
    assert (await application.ask(config, session.session_id, "start", model=parent)).status == "succeeded"
    renamed = configured(config, name="改名后的资料员")
    result = await application.ask(renamed, session.session_id, "continue", model=parent, callbacks=[CliProgress()])
    assert result.status == "succeeded", result.error
    assert human_text(worker.inputs[-1]) == ["first", "follow up"]
    output = capsys.readouterr().err
    assert "[SubAgent" not in output and "改名后的资料员" not in output
    assert "[Tool] Completed delegate_task (" in output
    assert any(row["payload"].get("subagent_name") == "改名后的资料员" for row in records(session) if row["event"] == "chain_start")


def test_persistent_denial_and_followup_in_separate_processes(config, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(path.read_text().replace('persistence = "per_invocation"', 'persistence = "per_thread"'))
    config = Config.load(path)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "evidence").write_text("private-process-detail-7319")
    external = tmp_path / "external"
    external.write_text("original")
    session = application.create_session(config, workdir=str(workdir))
    script = '''
import asyncio
import json
import sys
from rs_harness import application
from rs_harness.agents.main import build_agent
from rs_harness.config import Config
from tests.integration.test_subagents import call, delegate, request
from tests.models import ScriptedModel
from langchain_core.messages import AIMessage

async def run():
    config = Config.load(sys.argv[1])
    session_id, mode = sys.argv[2:4]
    if mode == "ask":
        parent = ScriptedModel([request(delegate("inspect and change", "first"), delegate("duplicate must not run", "duplicate")), AIMessage(content="parent complete")])
        worker = ScriptedModel([
            request(call("bash", {"command": "printf x >> marker"}, "effect"),
                    call("read_file", {"path": "evidence"}, "evidence")),
            request(call("write_file", {"path": "../external", "content": "changed"}, "external")),
            AIMessage(content="child denied"),
        ])
    else:
        parent = ScriptedModel(([request(delegate("recall previous evidence", "followup"))] if mode == "follow" else [])
                               + [AIMessage(content="parent complete")])
        worker = ScriptedModel([AIMessage(content="child complete")])
    def factory(config, **kwargs):
        kwargs.pop("model", None)
        return build_agent(config, model=parent, subagent_model=worker, **kwargs)
    application.build_agent = factory
    if mode in {"ask", "follow"}:
        result = await application.ask(config, session_id, "run")
    else:
        pending = await application.get_pending_approvals(config, session_id)
        if mode == "query":
            result = pending
        else:
            result = await application.resume(config, session_id, {
                item["id"]: {"decisions": [{"type": "approve"} for _ in item["action_requests"]]}
                for item in pending.interrupts
            })
    print(json.dumps({"result": result.model_dump(mode="json"),
                      "child_inputs": [[m.model_dump(mode="json") for m in batch] for batch in worker.inputs]}))
asyncio.run(run())
'''

    def run(mode):
        process = subprocess.run([sys.executable, "-c", script, str(path), session.session_id, mode],
                                 capture_output=True, text=True, timeout=30)
        assert process.returncode == 0, process.stderr + process.stdout
        assert process.stderr == ""
        return json.loads(process.stdout)

    first = run("ask")["result"]
    assert first["status"] == "succeeded", first
    assert not first["interrupts"]
    assert external.read_text() == "original"
    assert (workdir / "marker").read_text() == "x"
    queried = run("query")["result"]
    assert not queried["interrupts"]
    resumed = run("resume")
    assert resumed["result"]["error"]["type"] == "no_pending_approval"
    assert external.read_text() == "original"
    final = run("follow")
    assert final["result"]["status"] == "succeeded", final
    assert (workdir / "marker").read_text() == "x"
    messages = final["child_inputs"][0]
    assert [m["content"] for m in messages if m["type"] == "human"] == ["inspect and change", "recall previous evidence"]
    assert "private-process-detail-7319" in str(messages)
    actions = [m for m in messages if m["type"] == "tool"]
    assert len(actions) == 3
    assert actions[-1]["status"] == "error"
    assert sum(row["event"] == "tool_start" and row["payload"]["name"] == "bash" for row in records(session)) == 1
