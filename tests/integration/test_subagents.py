import asyncio
import json
import subprocess
import sys
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphInterrupt

from rs_harness import application
from rs_harness.agents.main import build_agent
from rs_harness.agents.middleware import StopHookMiddleware, build_middleware
from rs_harness.progress import CliProgress
from rs_harness.sessions.checkpoints import open_checkpointer
from rs_harness.sessions.recording import TrajectoryRecorder, TrajectoryWriter
from tests.images import write_test_image
from tests.models import ScriptedModel


def call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def request(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def delegate(query="Inspect the input", call_id="delegate", agent_id="worker"):
    return call("delegate_task", {"agent_id": agent_id, "query": query}, call_id)


def install_worker(monkeypatch, worker):
    def factory(config, **kwargs):
        return build_agent(config, subagent_model=worker, **kwargs)
    monkeypatch.setattr(application, "build_agent", factory)


def records(session):
    return [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]


def tool_messages(messages):
    return [message for message in messages if isinstance(message, ToolMessage)]


async def snapshot(config, session):
    async with open_checkpointer(config) as saver:
        graph = build_agent(config, workdir=session.meta.workdir, checkpointer=saver,
                            model=ScriptedModel([AIMessage(content="unused")]))
        return await graph.aget_state({"configurable": {"thread_id": session.session_id}})


def test_factories_use_separate_middleware_and_inherited_child_checkpointer(config, monkeypatch):
    graphs = []

    def capture(**kwargs):
        graphs.append(kwargs)
        return object()

    monkeypatch.setattr("langchain.agents.create_agent", capture)
    model = ScriptedModel([AIMessage(content="unused")])
    build_agent(config, model=model, workdir="/work", checkpointer="parent-saver")
    child, parent = graphs
    assert child.get("checkpointer") is None
    assert parent["checkpointer"] == "parent-saver"
    assert child["model"] is parent["model"] is model
    assert child["name"] == "rsh_worker"
    assert len(child["tools"]) == 6 and len(parent["tools"]) == 7
    assert "delegate_task" not in {tool.name for tool in child["tools"]}
    assert all(a is not b for a, b in zip(child["middleware"], [m for m in parent["middleware"] if m.name not in {"SubagentBatchMiddleware", "PermissionMiddleware", "MCPMiddleware"}], strict=True))
    assert not {"PermissionMiddleware", "MCPMiddleware"} & {m.name for m in child["middleware"]}
    assert {"PermissionMiddleware", "MCPMiddleware"} <= {m.name for m in parent["middleware"]}
    assert next(m for m in child["middleware"] if m.name == "ToolDenyMiddleware").allow_approval is False
    schema = parent["tools"][-1].tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"agent_id", "query"}


async def test_delegation_executes_tools_and_preserves_context_boundary(config, tmp_path, monkeypatch):
    (tmp_path / "input.txt").write_text("child-evidence")
    worker = ScriptedModel([
        request(call("read_file", {"path": "input.txt"}, "child-read")),
        AIMessage(content="Found child-evidence in input.txt"),
    ])
    parent = ScriptedModel([request(delegate("Read input.txt")), AIMessage(content="Verified")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "parent-private-context", model=parent)

    assert result.status == "succeeded", result.error
    assert [message.content for message in worker.inputs[0] if isinstance(message, HumanMessage)] == ["Read input.txt"]
    assert "parent-private-context" not in str(worker.inputs)
    assert "child-evidence" in tool_messages(worker.inputs[1])[0].content
    assert tool_messages(parent.inputs[1])[0].content == "Found child-evidence in input.txt"
    assert tool_messages(parent.inputs[1])[0].tool_call_id == "delegate"
    assert not any(message.tool_call_id == "child-read" for message in tool_messages(parent.inputs[1]))
    assert {tool.name for tool in parent.bound_tools} == {tool.name for tool in worker.bound_tools} | {"delegate_task"}
    assert len(parent.bound_tools) == 9 and len(worker.bound_tools) == 8


@pytest.mark.parametrize("content", ["", "x" * 70000, [{"type": "text", "text": "native block"}]])
async def test_final_content_is_returned_without_custom_validation_or_truncation(config, tmp_path, monkeypatch, content):
    worker = ScriptedModel([AIMessage(content=content)])
    parent = ScriptedModel([request(delegate()), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    assert tool_messages(parent.inputs[1])[0].content == content


@pytest.mark.parametrize("args, valid", [({}, False), ({"agent_id": "worker", "query": 42}, False), ({"agent_id": "worker", "query": "  "}, True)])
async def test_official_tool_schema(config, tmp_path, monkeypatch, args, valid):
    worker = ScriptedModel([AIMessage(content="worker")])
    parent = ScriptedModel([
        request(call("delegate_task", args, "invalid")), AIMessage(content="done"),
    ])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    assert worker.call_count == int(valid)
    assert tool_messages(parent.inputs[1])[0].status == ("success" if valid else "error")


async def test_new_delegations_isolate_history_and_todos(config, tmp_path, monkeypatch):
    main_todos = [{"content": "Main work", "status": "in_progress"}]
    child_todos = [{"content": "child-private-plan", "status": "completed"}]
    worker = ScriptedModel([
        request(call("write_todos", {"todos": child_todos}, "child-todo")),
        AIMessage(content="first result"), AIMessage(content="second result"),
    ])
    parent = ScriptedModel([
        request(call("write_todos", {"todos": main_todos}, "main-todo")),
        request(delegate("first task", "first")), AIMessage(content="one"),
        request(delegate("second task", "second")), AIMessage(content="two"),
    ])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    for text in ("start", "continue"):
        result = await application.ask(config, session.session_id, text, model=parent)
        assert result.status == "succeeded", result.error
    assert [m.content for m in worker.inputs[2] if not isinstance(m, SystemMessage)] == ["second task"]
    state = await snapshot(config, session)
    assert state.values["todos"] == main_todos
    assert "child-private-plan" not in str(state.values)
    assert {m.tool_call_id for m in tool_messages(state.values["messages"])} == {"main-todo", "first", "second"}


async def test_parallel_calls_to_same_child_graph_overlap(config, tmp_path, monkeypatch):
    entered = []
    ready = asyncio.Event()

    class ParallelWorker(ScriptedModel):
        async def _agenerate(self, messages, **kwargs):
            inputs = [m.content for m in messages if not isinstance(m, SystemMessage)]
            entered.append(inputs)
            if len(entered) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 5)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=inputs[0] + " result"))])

    worker = ParallelWorker([AIMessage(content="unused")])
    parent = ScriptedModel([request(delegate("alpha", "a"), delegate("beta", "b")), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "parallel", model=parent)
    assert result.status == "succeeded", result.error
    assert sorted(entered) == [["alpha"], ["beta"]]
    assert {m.tool_call_id: m.content for m in tool_messages(parent.inputs[1])} == {"a": "alpha result", "b": "beta result"}


@pytest.mark.parametrize("persistence", ["per_invocation", "per_thread"])
async def test_child_denial_then_followup_does_not_repeat_side_effect(config, tmp_path, monkeypatch, persistence):
    config = config.model_copy(update={"subagents": (config.subagents[0].model_copy(update={"persistence": persistence}),)})
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("original")
    worker = ScriptedModel([
        request(call("bash", {"command": "printf x >> marker.txt"}, "mark")),
        request(call("write_file", {"path": "../external.txt", "content": "changed"}, "external")),
        AIMessage(content="reported"), AIMessage(content="followed up"),
    ])
    parent = ScriptedModel([request(delegate()), AIMessage(content="done"),
                            request(delegate("followup", "followup")), AIMessage(content="done again")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(workdir))
    first = await application.ask(config, session.session_id, "run", model=parent)
    assert first.status == "succeeded" and not first.interrupts, first.error
    assert external.read_text() == "original"
    assert (workdir / "marker.txt").read_text() == "x"
    assert tool_messages(worker.inputs[-1])[-1].status == "error"
    assert not (await application.get_pending_approvals(config, session.session_id, model=parent)).interrupts
    resumed = await application.resume(config, session.session_id, {}, model=parent)
    assert resumed.error["type"] == "no_pending_approval"
    # Observation records cannot supply the child's continued conversation.
    session.trajectory_path.write_text("broken observation\n")
    final = await application.ask(config, session.session_id, "continue", model=parent)
    assert final.status == "succeeded", final.error
    expected = ["Inspect the input", "followup"] if persistence == "per_thread" else ["followup"]
    assert [m.content for m in worker.inputs[-1] if isinstance(m, HumanMessage)] == expected
    assert external.read_text() == "original" and (workdir / "marker.txt").read_text() == "x"


async def test_parallel_child_denials_preserve_allowed_siblings(config, tmp_path, monkeypatch, capsys):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    worker = ScriptedModel([
        request(call("write_file", {"path": "../a", "content": "a"}, "a"),
                call("write_file", {"path": "inside", "content": "kept"}, "inside")),
        request(call("write_file", {"path": "../b", "content": "b"}, "b")),
        AIMessage(content="worker complete"), AIMessage(content="worker complete"),
    ])
    parent = ScriptedModel([request(delegate("one", "one"), delegate("two", "two")), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    progress = CliProgress()
    session = application.create_session(config, workdir=str(workdir))
    result = await application.ask(config, session.session_id, "parallel", model=parent, callbacks=[progress])
    assert result.status == "succeeded" and not result.interrupts, result.error
    assert (workdir / "inside").read_text() == "kept"
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    output = capsys.readouterr().err
    assert output.count("[Tool] Completed delegate_task (") == 2
    assert output.count("[Model]") == 2
    assert not any(marker in output for marker in ("[SubAgent", "[Approval]", "[Middleware]", "write_file"))
    assert not progress._hidden_runs and not progress._tools and not progress._middleware
    assert len(tool_messages(parent.inputs[-1])) == 2
    writes = [row for row in records(session) if row["event"] == "tool_start" and row["payload"]["name"] == "write_file"]
    assert len(writes) == 1 and writes[0]["payload"]["inputs"]["path"] == "inside"


async def test_child_denial_and_tool_error_can_be_handled(config, tmp_path, monkeypatch):
    worker = ScriptedModel([
        request(call("bash", {"command": "sudo touch forbidden"}, "deny"),
                call("read_file", {"path": "missing.txt"}, "missing")),
        AIMessage(content="Could not perform the requested actions"),
    ])
    parent = ScriptedModel([request(delegate()), AIMessage(content="reported")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    assert [m.status for m in tool_messages(worker.inputs[1])] == ["error", "error"]
    assert not (tmp_path / "forbidden").exists()


async def test_parent_approval_does_not_authorize_child_external_access(config, tmp_path, monkeypatch):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    worker = ScriptedModel([
        request(call("write_file", {"path": "../child.txt", "content": "child"}, "child-write")),
        AIMessage(content="worker done"),
    ])
    parent = ScriptedModel([
        request(delegate(), call("write_file", {"path": "../parent.txt", "content": "parent"}, "parent-write"),
                call("bash", {"command": "printf x >> marker"}, "sibling")),
        AIMessage(content="done"),
    ])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(workdir))
    first = await application.ask(config, session.session_id, "run", model=parent)
    assert first.status == "interrupted" and worker.call_count == 0
    assert not (workdir / "marker").exists()
    second = await application.resume(config, session.session_id, {
        first.interrupts[0]["id"]: {"decisions": [{"type": "approve"}]},
    }, model=parent)
    assert second.status == "succeeded" and worker.call_count == 2, second.error
    assert not second.interrupts and first.interaction_id == second.interaction_id
    assert (tmp_path / "parent.txt").read_text() == "parent"
    assert (workdir / "marker").read_text() == "x"
    assert not (tmp_path / "child.txt").exists()
    assert tool_messages(worker.inputs[-1])[-1].status == "error"


async def test_child_uses_tool_limits_from_context(config, tmp_path, monkeypatch):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={
        "max_tool_output_bytes": 16, "bash_timeout_s": 0.05,
    })})
    worker = ScriptedModel([
        request(call("bash", {"command": "printf 123456789012345678901234567890"}, "output")),
        request(call("bash", {"command": "sleep 30"}, "timeout")),
        AIMessage(content="worker done"),
    ])
    parent = ScriptedModel([request(delegate()), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    output, timeout = tool_messages(worker.inputs[-1])
    assert json.loads(output.content)["truncated"] is True
    assert len(json.loads(output.content)["stdout"].encode()) <= 16
    assert timeout.status == "error" and json.loads(timeout.content)["timed_out"] is True


async def test_child_image_history_is_not_copied_to_parent(config, tmp_path, monkeypatch):
    write_test_image(tmp_path / "image.png")
    worker = ScriptedModel([
        request(call("read_image", {"path": "image.png"}, "image")), AIMessage(content="red and blue"),
    ])
    parent = ScriptedModel([request(delegate("Look at image.png")), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    assert any(isinstance(m, HumanMessage) and isinstance(m.content, list)
               and any(block.get("type") == "image" for block in m.content) for m in worker.inputs[-1])
    assert not any(block.get("type") == "image" for m in parent.inputs[-1] for block in m.content_blocks)
    assert tool_messages(parent.inputs[-1])[0].content == "red and blue"


async def test_parent_and_child_stop_hooks_only_see_their_own_history(config, tmp_path, monkeypatch):
    seen = []

    def check(messages):
        seen.append([m.content for m in messages])
        if messages[-1].content in {"child first", "parent first"}:
            return "verify once more"
        return None

    def middleware(config, **kwargs):
        return [*build_middleware(config, **kwargs)[:-1], StopHookMiddleware(hooks=[check])]

    monkeypatch.setattr("rs_harness.agents.main.build_middleware", middleware)
    monkeypatch.setattr("rs_harness.agents.subagent.build_middleware", middleware)
    worker = ScriptedModel([AIMessage(content="child first"), AIMessage(content="child verified")])
    parent = ScriptedModel([request(delegate("child request")), AIMessage(content="parent first"), AIMessage(content="parent verified")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "parent private", model=parent)
    assert result.status == "succeeded" and result.answer == "parent verified"
    assert worker.call_count == 2 and parent.call_count == 3
    assert seen[0] == ["child request", "child first"]
    assert "parent private" not in str(seen[:2])
    assert "child first" not in str(seen[2:])


async def test_cancelled_child_bash_cleans_process_and_releases_lock(config, tmp_path, monkeypatch):
    from rs_harness.sessions.catalog import SessionBusy, SessionCatalog
    from tests.unit.test_shell_tool import assert_stopped

    worker = ScriptedModel([
        request(call("bash", {"command": "sleep 30 & echo $! > child.pid; wait"}, "shell")),
    ])
    parent = ScriptedModel([request(delegate())])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    task = asyncio.create_task(application.ask(config, session.session_id, "run", model=parent))
    try:
        async with asyncio.timeout(5):
            while not (tmp_path / "child.pid").exists():
                await asyncio.sleep(0.01)
        with pytest.raises(SessionBusy):
            SessionCatalog.close(config, session.session_id)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert_stopped(int((tmp_path / "child.pid").read_text()))
    assert SessionCatalog.close(config, session.session_id).meta.status == "closed"


@pytest.mark.parametrize("external", [False, True])
async def test_child_budget_failure_does_not_create_approval(config, tmp_path, monkeypatch, external):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 1})})
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    path = "../external" if external else "internal"
    worker = ScriptedModel([
        request(call("write_file", {"path": path, "content": "written"}, "write")), AIMessage(content="unreachable"),
    ])
    parent = ScriptedModel([request(delegate()), AIMessage(content="unreachable")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(workdir))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.error["type"] == "ModelCallLimitExceededError" and not result.interrupts
    assert worker.call_count == parent.call_count == 1
    assert not (tmp_path / "external").exists()
    assert (workdir / "internal").exists() is not external
    messages = tool_messages((await snapshot(config, session)).values["messages"])
    assert messages[0].status == "error" and "subagent_budget_exhausted" in messages[0].content


async def test_parent_budget_does_not_count_child_model_calls(config, tmp_path, monkeypatch):
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 1})})
    worker = ScriptedModel([AIMessage(content="child done")])
    parent = ScriptedModel([request(delegate()), AIMessage(content="unreachable")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.error["type"] == "ModelCallLimitExceededError"
    assert parent.call_count == worker.call_count == 1
    assert tool_messages((await snapshot(config, session)).values["messages"])[0].content == "child done"


async def test_official_run_tree_retains_hidden_child_todos(config, tmp_path, monkeypatch, capsys):
    worker = ScriptedModel([
        request(call("write_todos", {"todos": [{"content": "child plan", "status": "completed"}]}, "plan")),
        AIMessage(content="worker done"),
    ])
    parent = ScriptedModel([request(delegate("private task text")), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    progress = CliProgress()
    result = await application.ask(config, session.session_id, "parent context", model=parent, callbacks=[progress])
    assert result.status == "succeeded", result.error
    output = capsys.readouterr().err
    assert "[SubAgent" not in output and "[Todo]" not in output
    assert output.count("[Tool] Completed delegate_task (") == 1 and output.count("[Model]") == 2
    assert "private task text" not in output and "parent context" not in output
    assert not progress._hidden_runs
    rows = records(session)
    starts = {row["run_id"]: row for row in rows if row["event"].endswith("_start")}
    child_tool = next(row for row in rows if row["event"] == "tool_start" and row["payload"]["name"] == "write_todos")
    ancestors = []
    current = child_tool
    while current["parent_run_id"]:
        current = starts[current["parent_run_id"]]
        ancestors.append(current["payload"].get("name"))
    assert "delegate_task" in ancestors and "rsh_worker" in ancestors and "rsh_agent" in ancestors
    assert len({row["run_id"] for row in rows if row["event"] == "model_start"}) == 4
    assert any(row["payload"].get("subagent_id") == "worker" for row in rows if row["event"] == "chain_start")
    assert {row["interaction_id"] for row in rows} == {result.interaction_id}


@pytest.mark.parametrize("error", [RuntimeError("provider failed"), asyncio.CancelledError()])
async def test_child_model_errors_and_cancellation_propagate(config, tmp_path, monkeypatch, capsys, error):
    started = asyncio.Event()

    class FailingWorker(ScriptedModel):
        async def _agenerate(self, messages, **kwargs):
            if isinstance(error, asyncio.CancelledError):
                started.set()
                await asyncio.Event().wait()
            raise error

    install_worker(monkeypatch, FailingWorker([AIMessage(content="unused")]))
    parent = ScriptedModel([request(delegate()), AIMessage(content="unreachable")])
    session = application.create_session(config, workdir=str(tmp_path))
    progress = CliProgress()
    if isinstance(error, asyncio.CancelledError):
        task = asyncio.create_task(application.ask(config, session.session_id, "run", model=parent, callbacks=[progress]))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not any(row["event"].endswith("_error") for row in records(session))
    else:
        result = await application.ask(config, session.session_id, "run", model=parent, callbacks=[progress])
        assert result.status == "failed" and result.error["type"] == "RuntimeError"
    output = capsys.readouterr().err
    assert "provider failed" not in output and "[SubAgent" not in output
    assert output.count("[Tool] Running delegate_task\n") == 1
    assert "[Tool] Completed delegate_task" not in output
    assert output.count("[Tool] Failed delegate_task") == (0 if isinstance(error, asyncio.CancelledError) else 1)
    assert parent.call_count == 1
    assert not progress._hidden_runs and not progress._tools


@pytest.mark.parametrize("error", [GraphInterrupt(), asyncio.CancelledError()])
async def test_recording_control_flow_is_not_an_error(tmp_path, error):
    path = tmp_path / "trajectory.jsonl"
    recorder = TrajectoryRecorder(TrajectoryWriter(path, session_id="session", interaction_id="interaction"))
    for method in (recorder.on_chain_error, recorder.on_llm_error, recorder.on_tool_error):
        await method(error, run_id=uuid4())
    assert not path.exists()


def test_cli_json_child_denial_and_no_pending_in_another_process(config, tmp_path):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("original")
    session = application.create_session(config, workdir=str(workdir))
    script = '''
import sys
from langchain_core.messages import AIMessage
from rs_harness import application, cli
from rs_harness.agents.main import build_agent
from tests.models import ScriptedModel
mode = sys.argv.pop(1)
def call(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}
parent = ScriptedModel([AIMessage(content="", tool_calls=[call("delegate_task", {"agent_id":"worker", "query":"work"}, "delegate")]), AIMessage(content="parent complete")]
                       if mode == "ask" else [AIMessage(content="parent complete")])
worker = ScriptedModel([AIMessage(content="", tool_calls=[call("write_file", {"path":"../external.txt", "content":"changed"}, "external")]), AIMessage(content="child denied")]
                       if mode == "ask" else [AIMessage(content="child complete")])
def factory(config, **kwargs):
    kwargs.pop("model", None)
    return build_agent(config, model=parent, subagent_model=worker, **kwargs)
application.build_agent = factory
if mode == "ask":
    cli.main()
else:
    import asyncio
    import json
    from pathlib import Path
    from rs_harness.config import Config
    config = Config.load(sys.argv[sys.argv.index("--config") + 1])
    session_id = sys.argv[sys.argv.index("--session-id") + 1]
    if mode == "query":
        result = asyncio.run(application.get_pending_approvals(config, session_id))
    else:
        decisions = json.loads(Path(sys.argv[sys.argv.index("--decisions-file") + 1]).read_text())
        result = asyncio.run(application.resume(config, session_id, decisions))
    print(result.model_dump_json())
'''

    def run(mode, *args, expected=0):
        process = subprocess.run(
            [sys.executable, "-c", script, mode, *args, "--config", str(tmp_path / "config.toml"),
             "--session-id", session.session_id, "--json"], capture_output=True, text=True, timeout=30,
        )
        assert process.returncode == expected, process.stderr + process.stdout
        assert process.stderr == ""
        return json.loads(process.stdout)

    first = run("ask", "ask", "delegate")
    assert first["status"] == "succeeded" and not first["interrupts"]
    assert external.read_text() == "original"
    queried = run("query", "approvals")
    assert not queried["interrupts"]
    decisions = tmp_path / "decisions.json"
    decisions.write_text("{}")
    final = run("resume", "resume", "--decisions-file", str(decisions))
    assert final["error"]["type"] == "no_pending_approval" and external.read_text() == "original"
    assert sum(row["event"] == "user_input" for row in records(session)) == 1
