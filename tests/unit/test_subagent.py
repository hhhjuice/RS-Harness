from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from threading import Event
from pathlib import Path
from types import SimpleNamespace

import pytest

from rs_harness.agent_loop import InputRequest, create_session, run
from rs_harness.config import Config
from rs_harness.runtime.hooks import HookRegistry, PermissionRequest
from rs_harness.runtime.session import Session

SUB_PROMPT_TEXT = (
    "sub-system\nWorking directory: {workdir}\n"
    "Complete the task, then return a concise final answer."
)


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(kwargs)
        return response


def _config(tmp_path: Path, **loop: object) -> Config:
    prompt = tmp_path / "system.md"
    prompt.write_text(
        "system\nWorking directory: {workdir}\n"
        "Only create, modify, or edit files inside this directory.",
        encoding="utf-8",
    )
    sub_prompt = tmp_path / "subagent.md"
    sub_prompt.write_text(SUB_PROMPT_TEXT, encoding="utf-8")
    kwargs: dict[str, object] = {
        "model": {"model": "test-model", "api_key_env": "UNUSED_RS_KEY"},
        "runtime": {
            "sessions_dir": tmp_path / "sessions",
            "system_prompt_file": prompt,
            "subagent_system_prompt_file": sub_prompt,
            "dotenv_file": tmp_path / "missing.env",
        },
    }
    if loop:
        kwargs["loop"] = dict(loop)
    return Config(**kwargs)  # type: ignore[arg-type]


def _response(content=None, calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=calls or [], role="assistant")
            )
        ]
    )


def _call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _sub_sessions(config: Config, parent_id: str) -> list[Session]:
    return [
        session
        for session in (Session.open(config, state.session_id) for state in Session.list(config))
        if session.state.parent_session_id == parent_id
    ]


def _task_result(output: str) -> tuple[dict[str, str], str]:
    header, body = output.split("\n\n", 1)
    return dict(line.split(": ", 1) for line in header.splitlines()), body


def _parent_task(config, parent_id, arguments, child_responses=(), **kwargs):
    # Existing synchronous unit scenarios opt out explicitly; async behavior
    # is covered by dedicated tests below and is the production default.
    arguments = dict(arguments)
    arguments.setdefault("background", False)
    client = SequenceClient([
        _response(calls=[_call("task-call", "task", json.dumps(arguments))]),
        *child_responses,
        _response("parent done"),
    ])
    result = run(parent_id, InputRequest(query="delegate"), config, openai_sdk_client=client, **kwargs)
    assert result.status == "succeeded"
    output = [m for m in Session.open(config, parent_id).messages() if m["role"] == "tool"][-1]["content"]
    return output, client


def test_task_delegates_to_child_session_and_returns_final_text_only(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "one.txt").write_text("one", encoding="utf-8")
    (workspace / "two.txt").write_text("two", encoding="utf-8")
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"Count the .txt files"}')]),
            _response(calls=[_call("g1", "glob", '{"pattern":"*.txt"}')]),
            _response("found 2 text files"),
            _response("There are 2 text files."),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id

    result = run(
        parent_id, InputRequest(query="delegate the count"), config, openai_sdk_client=client
    )

    assert result.status == "succeeded"
    assert result.answer == "There are 2 text files."
    # Parent request exposes task; the subagent request hides it and starts
    # from its own system prompt and the delegation prompt only.
    assert [tool["function"]["name"] for tool in client.calls[0]["tools"]].count("task") == 1
    assert all(tool["function"]["name"] != "task" for tool in client.calls[1]["tools"])
    assert client.calls[1]["messages"][0] == {
        "role": "system",
        "content": SUB_PROMPT_TEXT.replace("{workdir}", str(workspace.resolve())) + "\n\nSubagent name: 目录检查员",
    }
    assert client.calls[1]["messages"][1] == {"role": "user", "content": "Count the .txt files"}
    # The subagent observed the real workspace through its own tool call.
    assert client.calls[2]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "g1",
        "content": "one.txt\ntwo.txt",
    }
    # The parent saw only the final text as the task observation.
    parent_tool = client.calls[3]["messages"][-1]
    assert parent_tool["role"] == "tool" and parent_tool["tool_call_id"] == "t1"
    metadata, body = _task_result(parent_tool["content"])
    assert body == "found 2 text files"
    assert metadata["name"] == "目录检查员"
    assert metadata["status"] == "succeeded"
    assert metadata["session_status"] == "running"

    reopened = Session.open(config, parent_id)
    roles = [message["role"] for message in reopened.messages()]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]

    subs = _sub_sessions(config, parent_id)
    assert len(subs) == 1
    sub = subs[0]
    assert sub.state.status == "running"
    assert sub.state.name == "目录检查员"
    assert metadata["session_id"] == sub.state.session_id
    assert sub.state.workdir == str(workspace.resolve())
    assert sub.state.turn == 2
    assert [message["role"] for message in sub.messages()] == [
        "system", "user", "assistant", "tool", "assistant",
    ]


def test_subagent_messages_never_enter_the_parent_ledger(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"explore"}')]),
            _response(calls=[_call("g1", "glob", '{"pattern":"*"}')]),
            _response(calls=[_call("w1", "write_file", '{"path":"out.txt","content":"data"}')]),
            _response("done exploring"),
            _response("parent summary"),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id

    run(parent_id, InputRequest(query="go"), config, openai_sdk_client=client)

    ledger = (session.path / "messages.jsonl").read_text(encoding="utf-8")
    assert "done exploring" in ledger
    assert '"glob"' not in ledger and '"write_file"' not in ledger
    parent_messages = Session.open(config, parent_id).messages()
    assert len([m for m in parent_messages if m["role"] == "tool"]) == 1
    # The subagent's write is visible in the shared workspace.
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "data"


def test_failed_subagent_becomes_an_observation_the_parent_recovers_from(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"do it"}')]),
            _response(None),  # empty response: no text, no tool calls
            _response("recovered without the subagent"),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id

    result = run(parent_id, InputRequest(query="delegate"), config, openai_sdk_client=client)

    assert result.status == "succeeded"
    assert result.answer == "recovered without the subagent"
    parent_tool = [
        m for m in Session.open(config, parent_id).messages() if m["role"] == "tool"
    ][0]
    assert parent_tool["tool_call_id"] == "t1"
    assert "empty_model_response" in parent_tool["content"]
    sub = _sub_sessions(config, parent_id)[0]
    assert sub.state.status == "running"
    assert sub.state.errors[0].code == "empty_model_response"


def test_subagent_budget_exhaustion_is_reported_not_raised(tmp_path: Path):
    config = _config(tmp_path, max_turns=3)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"loop forever"}')]),
            _response(calls=[_call("g1", "glob", '{"pattern":"*"}')]),
            _response(calls=[_call("g2", "glob", '{"pattern":"*"}')]),
            _response(calls=[_call("g3", "glob", '{"pattern":"*"}')]),
            _response("gave up on the subagent"),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id

    result = run(parent_id, InputRequest(query="delegate"), config, openai_sdk_client=client)

    assert result.status == "succeeded"
    parent_tool = [
        m for m in Session.open(config, parent_id).messages() if m["role"] == "tool"
    ][0]
    assert "budget_exceeded" in parent_tool["content"]
    sub = _sub_sessions(config, parent_id)[0]
    assert sub.state.status == "running"


def test_permission_callback_is_forwarded_to_the_subagent_loop(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    requests: list[PermissionRequest] = []
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"delete things"}')]),
            _response(calls=[_call("b1", "bash", '{"command":"rm notes.txt"}')]),
            _response("subagent finished"),
            _response("parent done"),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id

    result = run(
        parent_id,
        InputRequest(query="delegate"),
        config,
        openai_sdk_client=client,
        permission_callback=lambda request: requests.append(request) or False,
    )

    assert result.status == "succeeded"
    assert [request.tool_name for request in requests] == ["bash"]
    assert requests[0].arguments == {"command": "rm notes.txt"}
    # Denial became the subagent's tool observation, not an exception.
    sub_tool = [
        m for m in _sub_sessions(config, parent_id)[0].messages() if m["role"] == "tool"
    ][0]
    assert sub_tool["tool_call_id"] == "b1"
    assert "Permission denied: Potentially destructive command" in sub_tool["content"]


def test_subagent_cannot_delegate_again_even_if_the_model_tries(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"outer"}')]),
            _response(calls=[_call("t2", "task", '{"name":"目录检查员","background":false,"prompt":"inner"}')]),
            _response("inner refused, finished"),
            _response("parent done"),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id

    run(parent_id, InputRequest(query="delegate"), config, openai_sdk_client=client)

    sub_tool = [
        m for m in _sub_sessions(config, parent_id)[0].messages() if m["role"] == "tool"
    ][0]
    assert sub_tool["tool_call_id"] == "t2"
    assert "requires an active parent agent loop" in sub_tool["content"]
    assert len(_sub_sessions(config, parent_id)) == 1


def test_missing_subagent_prompt_file_degrades_to_an_observation(tmp_path: Path):
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"subagent_system_prompt_file": tmp_path / "missing-sub.md"}
            )
        }
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"anything"}')]),
            _response("parent continues"),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id

    result = run(parent_id, InputRequest(query="delegate"), config, openai_sdk_client=client)

    assert result.status == "succeeded"
    parent_tool = [
        m for m in Session.open(config, parent_id).messages() if m["role"] == "tool"
    ][0]
    assert "Error: Unable to open or create subagent" in parent_tool["content"]
    assert _sub_sessions(config, parent_id) == []


def test_subagent_notices_are_prefixed_and_todos_stay_isolated(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"plan and finish"}')]),
            _response(
                calls=[
                    _call(
                        "todo-1",
                        "todo_write",
                        '{"todos":[{"content":"Step","status":"completed"}]}',
                    )
                ]
            ),
            _response("subagent done"),
            _response("parent done"),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id
    notices: list[str] = []

    run(
        parent_id,
        InputRequest(query="delegate"),
        config,
        openai_sdk_client=client,
        console=lambda message: notices.append(message),
    )

    assert any(n.startswith("[sub:目录检查员|") and "[HOOK] UserPromptSubmit: working in" in n for n in notices)
    sub_todo_notice = next(
        (n for n in notices if "## Current Tasks" in n and n.startswith("[sub:目录检查员|")), None
    )
    assert sub_todo_notice is not None and "[x] Step" in sub_todo_notice
    assert any(n.startswith("[HOOK] task(") for n in notices)
    # Planning state stays in the subagent's session, not the parent's.
    parent = Session.open(config, parent_id)
    assert parent.state.todos == []
    assert _sub_sessions(config, parent_id)[0].state.todos != []


def test_subagent_events_go_to_their_own_ledger(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"explore"}')]),
            _response(calls=[_call("g1", "glob", '{"pattern":"*"}')]),
            _response("done"),
            _response("parent done"),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id

    run(parent_id, InputRequest(query="delegate"), config, openai_sdk_client=client)

    parent_events = [
        json.loads(line)
        for line in (session.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # The parent ledger audits the task call itself, not the subagent's tools.
    assert all(event.get("tool_name") != "glob" for event in parent_events)
    assert any(event.get("tool_name") == "task" for event in parent_events)

    sub = _sub_sessions(config, parent_id)[0]
    sub_events = [
        json.loads(line)
        for line in (sub.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("tool_name") == "glob" for event in sub_events)


def test_custom_hooks_run_inside_the_subagent_loop(tmp_path: Path, monkeypatch):
    import rs_harness.agent_loop as agent_loop_module

    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    seen: list[str] = []
    custom = HookRegistry()
    custom.register(
        "PreToolUse",
        lambda context: seen.append(f"{context.session_id}:{context.tool_name}") or None,
        name="track_hook",
    )
    client = SequenceClient(
        [
            _response(calls=[_call("t1", "task", '{"name":"目录检查员","background":false,"prompt":"explore"}')]),
            _response(calls=[_call("g1", "glob", '{"pattern":"*"}')]),
            _response("done"),
            _response("parent done"),
        ]
    )
    session = create_session(config, workdir=workspace)
    parent_id = session.state.session_id
    run_order: list[str] = []
    original = agent_loop_module.run

    def tracking_run(session_id, request, config, **kwargs):
        run_order.append(session_id)
        return original(session_id, request, config, **kwargs)

    monkeypatch.setattr(agent_loop_module, "run", tracking_run)

    result = original(
        parent_id, InputRequest(query="delegate"), config,
        openai_sdk_client=client, hooks=custom,
    )

    assert result.status == "succeeded"
    # The parent ran via the direct call above; only the nested subagent run
    # passes through the patched entry point.
    assert len(run_order) == 1
    sub_id = run_order[0]
    assert sub_id != parent_id
    # The custom hook fired in both loops with the right session identity.
    assert f"{parent_id}:task" in seen
    assert f"{sub_id}:glob" in seen
    assert f"{parent_id}:glob" not in seen


def test_same_child_continues_with_history_and_todos_across_parent_turns(tmp_path: Path):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    (tmp_path / "fact.txt").write_text("EPSG:4326", encoding="utf-8")
    child_id = None

    def follow_up(kwargs):
        nonlocal child_id
        metadata, body = _task_result(kwargs["messages"][-1]["content"])
        child_id = metadata["session_id"]
        assert body == "inspection ready"
        assert "EPSG:4326" not in json.dumps(kwargs["messages"])
        return _response(calls=[_call("follow-up", "task", json.dumps({
            "session_id": child_id, "background": False, "prompt": "What coordinate system did you find?",
        }))])

    def answer_from_history(kwargs):
        messages = kwargs["messages"]
        assert sum(m["role"] == "system" for m in messages) == 1
        assert messages[-1]["content"] == "What coordinate system did you find?"
        assert any(m["role"] == "tool" and m["content"] == "EPSG:4326" for m in messages)
        assert Session.open(config, child_id).state.todos[0].content == "Inspect imagery"
        return _response("EPSG:4326")

    client = SequenceClient([
        _response(calls=[_call("create", "task", '{"name":"影像检查员","background":false,"prompt":"Inspect the imagery"}')]),
        _response(calls=[
            _call("read", "read_file", '{"path":"fact.txt"}'),
            _call("todo", "todo_write", '{"todos":[{"content":"Inspect imagery","status":"in_progress"}]}'),
        ]),
        _response("inspection ready"),
        follow_up,
        answer_from_history,
        _response("parent done"),
    ])
    result = run(parent.state.session_id, InputRequest(query="delegate and follow up"), config, openai_sdk_client=client)
    assert result.status == "succeeded"
    child = Session.open(config, child_id)
    assert child.state.name == "影像检查员"
    assert child.state.status == "running"
    assert len(_sub_sessions(config, parent.state.session_id)) == 1
    assert [m["content"] for m in child.messages() if m["role"] == "user"] == [
        "Inspect the imagery", "What coordinate system did you find?",
    ]
    config.runtime.subagent_system_prompt_file.unlink()
    output, next_client = _parent_task(config, parent.state.session_id, {
        "session_id": child_id, "prompt": "Continue the same inspection",
    }, [_response("still working")])
    assert _task_result(output)[0]["session_id"] == child_id
    assert next_client.calls[1]["messages"][0] == child.messages()[0]
    assert next_client.calls[1]["messages"][-2]["content"] == "EPSG:4326"
    assert Session.open(config, child_id).state.todos == child.state.todos


def test_subagent_restores_identity_and_context_in_a_new_process(tmp_path: Path):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    output, _ = _parent_task(config, parent.state.session_id, {
        "name": "跨进程检查员", "prompt": "Remember the inspected projection",
    }, [_response("remembered EPSG:3857")])
    child_id = _task_result(output)[0]["session_id"]
    config.runtime.subagent_system_prompt_file.unlink()
    config_file = tmp_path / "config.json"
    config_file.write_text(config.model_dump_json(), encoding="utf-8")
    script = textwrap.dedent('''
        import json
        import runpy
        import sys
        from pathlib import Path
        from rs_harness.config import Config
        from rs_harness.runtime.session import Session
        helper = runpy.run_path(sys.argv[1])
        config = Config.model_validate_json(Path(sys.argv[2]).read_text())
        parent = Session.open(config, sys.argv[3])
        previous = [m for m in parent.messages() if m['role'] == 'tool'][-1]
        metadata, _ = helper['_task_result'](previous['content'])
        def reply(kwargs):
            messages = kwargs['messages']
            assert messages[-2]['content'] == 'remembered EPSG:3857'
            assert messages[-1]['content'] == 'Continue after restart'
            assert '跨进程检查员' in messages[0]['content']
            assert sum(m['role'] == 'system' for m in messages) == 1
            return helper['_response']('resumed EPSG:3857')
        output, _ = helper['_parent_task'](config, parent.state.session_id, {
            'session_id': metadata['session_id'], 'prompt': 'Continue after restart',
        }, [reply])
        print(json.dumps(helper['_task_result'](output), ensure_ascii=False))
    ''')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(config_file), parent.state.session_id],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True, timeout=20,
    )
    metadata, body = json.loads(completed.stdout)
    assert metadata["session_id"] == child_id
    assert metadata["name"] == "跨进程检查员"
    assert body == "resumed EPSG:3857"
    assert len(_sub_sessions(config, parent.state.session_id)) == 1


def test_task_close_is_idempotent_and_cannot_close_a_sibling_by_name(tmp_path: Path):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    ids = []
    for answer in ("first child", "second child"):
        output, _ = _parent_task(config, parent.state.session_id, {
            "name": "影像检查员", "prompt": "inspect",
        }, [_response(answer)])
        ids.append(_task_result(output)[0]["session_id"])
    assert ids[0] != ids[1]
    events = []
    hooks = HookRegistry()
    for event in ("PreToolUse", "PostToolUse"):
        hooks.register(event, lambda context: events.append((context.event, context.arguments["action"])))
    child = Session.open(config, ids[0])
    original_messages = child.messages()
    for _ in range(2):
        output, client = _parent_task(config, parent.state.session_id, {
            "action": "close", "session_id": ids[0],
        }, hooks=hooks)
        metadata, _ = _task_result(output)
        assert metadata["session_status"] == "closed" and metadata["status"] == "succeeded"
        assert metadata["name"] == "影像检查员"
        assert len(client.calls) == 2  # both requests belong to the parent
    assert events == [("PreToolUse", "close"), ("PostToolUse", "close")] * 2
    assert Session.open(config, ids[0]).messages() == original_messages
    assert Session.open(config, ids[1]).state.status == "running"
    output, client = _parent_task(config, parent.state.session_id, {
        "session_id": ids[0], "prompt": "too late",
    })
    assert "session_closed" in output and len(client.calls) == 2
    # Continuing the other identically named child remains possible.
    output, _ = _parent_task(config, parent.state.session_id, {
        "session_id": ids[1], "prompt": "continue second",
    }, [_response("second continued")])
    assert _task_result(output)[1] == "second continued"


@pytest.mark.parametrize("failure", ["empty", "timeout", "budget"])
def test_failed_interaction_can_resume_the_same_child(tmp_path: Path, failure: str):
    config = _config(tmp_path, max_turns=2)
    parent = create_session(config, workdir=tmp_path)
    responses = {
        "empty": [_response(None)],
        "timeout": [TimeoutError("synthetic timeout")],
        "budget": [
            _response(calls=[_call("g1", "glob", '{"pattern":"*.txt"}')]),
            _response(calls=[_call("g2", "glob", '{"pattern":"*.txt"}')]),
        ],
    }[failure]
    output, _ = _parent_task(config, parent.state.session_id, {
        "name": "可恢复检查员", "prompt": "inspect",
    }, responses)
    metadata, _ = _task_result(output)
    child_id = metadata["session_id"]
    assert metadata["status"] == "failed" and metadata["session_status"] == "running"
    output, _ = _parent_task(config, parent.state.session_id, {
        "session_id": child_id, "prompt": "try the next step",
    }, [_response("recovered")])
    assert _task_result(output)[1] == "recovered"
    child = Session.open(config, child_id)
    assert len(child.state.errors) == 1
    assert child.state.status == "running"
    if failure == "budget":
        assert child.state.turn == 3  # each interaction receives a fresh budget of two


@pytest.mark.parametrize("action", ["run", "close"])
def test_task_rejects_foreign_sessions_and_workdir_without_modifying_them(tmp_path: Path, action: str):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    other_parent = create_session(config, workdir=tmp_path)
    foreign = Session.create(config, workdir=tmp_path, parent_session_id=other_parent.state.session_id)
    other_workspace = tmp_path / "other-workspace"
    other_workspace.mkdir()
    mismatch = Session.create(config, workdir=other_workspace, parent_session_id=parent.state.session_id)
    for target in (parent, other_parent, foreign, mismatch):
        state_before = (target.path / "session.json").read_text()
        messages_before = (target.path / "messages.jsonl").read_text()
        arguments = {"action": action, "session_id": target.state.session_id}
        if action == "run":
            arguments["prompt"] = "must refuse"
        output, client = _parent_task(config, parent.state.session_id, arguments)
        assert output.startswith("Error:") and "session_status:" not in output
        assert len(client.calls) == 2
        if target is not parent:  # only the requesting parent's own interaction is appended
            assert (target.path / "session.json").read_text() == state_before
            assert (target.path / "messages.jsonl").read_text() == messages_before
    assert len(Session.list(config)) == 4


@pytest.mark.parametrize("status", ["closed", "failed"])
def test_old_terminal_child_is_not_reactivated(tmp_path: Path, status: str):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    child = Session.create(config, workdir=tmp_path, parent_session_id=parent.state.session_id)
    child.close(status)
    before = (child.path / "session.json").read_text()
    output, client = _parent_task(config, parent.state.session_id, {
        "session_id": child.state.session_id, "prompt": "resume old session",
    })
    metadata, body = _task_result(output)
    assert metadata["name"] == "未命名子代理"
    assert metadata["status"] == "failed" and metadata["session_status"] == status
    assert "session_closed" in body
    assert len(client.calls) == 2
    assert (child.path / "session.json").read_text() == before


def test_unmatched_child_tool_call_is_not_replayed(tmp_path: Path):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    child = Session.create(config, workdir=tmp_path, parent_session_id=parent.state.session_id)
    with child.locked():
        child.append({"role": "assistant", "tool_calls": [{
            "id": "unfinished", "type": "function", "function": {
                "name": "write_file", "arguments": '{"path":"must-not-exist.txt","content":"side effect"}',
            },
        }]})
    before = child.messages()
    output, client = _parent_task(config, parent.state.session_id, {
        "session_id": child.state.session_id, "prompt": "resume",
    })
    assert "unmatched_tool_call" in output
    assert len(client.calls) == 2
    assert Session.open(config, child.state.session_id).messages() == before
    assert not (tmp_path / "must-not-exist.txt").exists()


def test_name_is_redacted_before_state_ledgers_console_and_model_history(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    secret = "synthetic-subagent-key"
    monkeypatch.setenv("UNUSED_RS_KEY", secret)
    parent = create_session(config, workdir=tmp_path)
    notices = []
    output, client = _parent_task(config, parent.state.session_id, {
        "name": f"检查员 {secret}", "prompt": "inspect",
    }, [_response("done")], console=notices.append)
    metadata, _ = _task_result(output)
    assert metadata["name"] == "检查员 [REDACTED]"
    child = Session.open(config, metadata["session_id"])
    assert child.state.name == "检查员 [REDACTED]"
    assert "Subagent name: 检查员 [REDACTED]" in child.messages()[0]["content"]
    assert secret not in output and secret not in "\n".join(notices)
    assert secret not in repr(client.calls)
    for file in config.runtime.sessions_dir.rglob("*"):
        if file.suffix in {".json", ".jsonl"}:
            assert secret not in file.read_text()


def test_legacy_running_child_keeps_no_name_and_console_escapes_delimiters(tmp_path: Path):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    legacy = Session.create(config, workdir=tmp_path, parent_session_id=parent.state.session_id)
    output, _ = _parent_task(config, parent.state.session_id, {
        "session_id": legacy.state.session_id, "prompt": "continue legacy",
    }, [_response("legacy done")])
    assert _task_result(output)[0]["name"] == "未命名子代理"
    assert Session.open(config, legacy.state.session_id).state.name is None
    notices = []
    output, _ = _parent_task(config, parent.state.session_id, {
        "name": "检查|员]", "prompt": "inspect",
    }, [_response("done")], console=notices.append)
    assert _task_result(output)[0]["name"] == "检查|员]"
    assert any(line.startswith("[sub:检查\\|员\\]|") for line in notices)


@pytest.mark.parametrize("action", ["run", "close"])
def test_unknown_id_never_creates_a_replacement_child(tmp_path: Path, action: str):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    arguments = {"action": action, "session_id": "missing-child"}
    if action == "run":
        arguments["prompt"] = "continue"
    output, client = _parent_task(config, parent.state.session_id, arguments)
    assert output.startswith("Error:") and "Session not found" in output
    assert len(client.calls) == 2
    assert [state.session_id for state in Session.list(config)] == [parent.state.session_id]


def test_workspace_approval_is_not_reused_by_persistent_child(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("outside content", encoding="utf-8")
    parent = create_session(config, workdir=workspace)
    requests = []

    def permit_once(request):
        requests.append(request)
        return len(requests) == 1

    read = _response(calls=[_call("read-outside", "read_file", '{"path":"../outside.txt"}')])
    output, _ = _parent_task(config, parent.state.session_id, {
        "name": "权限检查员", "prompt": "read outside",
    }, [read, _response("first read complete")], permission_callback=permit_once)
    child_id = _task_result(output)[0]["session_id"]
    _parent_task(config, parent.state.session_id, {
        "session_id": child_id, "prompt": "read again",
    }, [read, _response("second read refused")], permission_callback=permit_once)
    assert len(requests) == 2
    observations = [m["content"] for m in Session.open(config, child_id).messages() if m["role"] == "tool"]
    assert observations[0] == "outside content"
    assert observations[1].startswith("Permission denied:")


def test_default_task_launches_multiple_children_without_blocking_parent(tmp_path: Path):
    config = _config(tmp_path)
    config = config.model_copy(update={
        "runtime": config.runtime.model_copy(update={"max_background_subagents": 2})
    })
    parent = create_session(config, workdir=tmp_path)
    child_started = Event()
    release_children = Event()
    parent_continued = Event()

    class ConcurrentClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.parent_calls = 0
            self.child_calls = 0
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            names = [tool["function"]["name"] for tool in kwargs["tools"]]
            if "task" in names:
                self.parent_calls += 1
                if self.parent_calls == 1:
                    return _response(calls=[
                        _call("a", "task", json.dumps({"name": "第一检查员", "prompt": "first"})),
                        _call("b", "task", json.dumps({"name": "第二检查员", "prompt": "second"})),
                    ])
                parent_continued.set()
                return _response("parent continued while children ran")
            self.child_calls += 1
            child_started.set()
            assert release_children.wait(5), "children were not released"
            return _response(f"child {self.child_calls} complete")

    client = ConcurrentClient()
    result = run(parent.state.session_id, InputRequest(query="launch two"), config, openai_sdk_client=client)
    assert result.status == "succeeded"
    assert result.answer == "parent continued while children ran"
    assert parent_continued.is_set()
    assert child_started.is_set()
    children = _sub_sessions(config, parent.state.session_id)
    assert len(children) == 2
    assert all(child.state.execution_status == "running" for child in children)
    release_children.set()
    deadline = __import__("time").monotonic() + 5
    while __import__("time").monotonic() < deadline:
        children = [Session.open(config, child.state.session_id) for child in children]
        if all(child.state.execution_status == "succeeded" for child in children):
            break
        __import__("time").sleep(0.01)
    assert all(child.state.execution_status == "succeeded" for child in children)
    assert all(child.state.status == "running" for child in children)


def test_status_and_wait_retrieve_background_result(tmp_path: Path):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    release = Event()
    started = Event()
    child_id = None

    class StatusClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            has_task = any(tool["function"]["name"] == "task" for tool in kwargs["tools"])
            if has_task and self.calls == 1:
                return _response(calls=[_call("create", "task", json.dumps({"name": "异步检查员", "prompt": "inspect"}))])
            if has_task and self.calls == 2:
                nonlocal child_id
                child_id = _task_result(kwargs["messages"][-1]["content"])[0]["session_id"]
                return _response(calls=[_call("status", "task", json.dumps({"action": "status", "session_id": child_id}))])
            if has_task and self.calls == 3:
                return _response(calls=[_call("wait", "task", json.dumps({"action": "wait", "session_id": child_id, "timeout_ms": 1}))])
            return _response("parent observed background result")

    class ChildClient(StatusClient):
        def create(self, **kwargs):
            has_task = any(tool["function"]["name"] == "task" for tool in kwargs["tools"])
            if not has_task:
                started.set()
                assert release.wait(5)
                return _response("background answer")
            return super().create(**kwargs)

    client = ChildClient()
    first = run(parent.state.session_id, InputRequest(query="start"), config, openai_sdk_client=client)
    assert first.status == "succeeded"
    assert started.wait(5)
    # A separate parent interaction can inspect the same child while its worker is running.
    second = run(parent.state.session_id, InputRequest(query="poll"), config, openai_sdk_client=client)
    assert second.status == "succeeded"
    release.set()
    deadline = __import__("time").monotonic() + 5
    while __import__("time").monotonic() < deadline:
        state = Session.open(config, child_id).state
        if state.execution_status == "succeeded":
            break
        __import__("time").sleep(0.01)
    assert Session.open(config, child_id).state.execution_answer == "background answer"


def test_running_background_child_can_be_cooperatively_cancelled(tmp_path: Path):
    config = _config(tmp_path)
    parent = create_session(config, workdir=tmp_path)
    started = Event()
    release = Event()
    child_id = None

    class CancelClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            has_task = any(tool["function"]["name"] == "task" for tool in kwargs["tools"])
            latest_user = [m for m in kwargs["messages"] if m["role"] == "user"][-1]["content"]
            if has_task and self.calls == 1:
                return _response(calls=[_call("create", "task", json.dumps({"name": "可取消检查员", "prompt": "long"}))])
            if has_task and latest_user == "start":
                return _response("parent started child")
            if has_task and latest_user == "cancel":
                return _response(calls=[_call("close", "task", json.dumps({"action": "close", "session_id": child_id}))])
            if has_task:
                return _response("parent cancellation requested")
            started.set()
            assert release.wait(5)
            return _response("child answer after cancellation")

    client = CancelClient()
    first = run(parent.state.session_id, InputRequest(query="start"), config, openai_sdk_client=client)
    assert first.status == "succeeded"
    child_id = _task_result([m for m in Session.open(config, parent.state.session_id).messages() if m["role"] == "tool"][-1]["content"])[0]["session_id"]
    assert started.wait(5)
    second = run(parent.state.session_id, InputRequest(query="cancel"), config, openai_sdk_client=client)
    assert second.status == "succeeded", second
    assert "cancel_requested" in [m["content"] for m in Session.open(config, parent.state.session_id).messages() if m["role"] == "tool"][-1]
    release.set()
    deadline = __import__("time").monotonic() + 5
    while __import__("time").monotonic() < deadline:
        state = Session.open(config, child_id).state
        if state.execution_status == "cancelled":
            break
        __import__("time").sleep(0.01)
    state = Session.open(config, child_id).state
    assert state.execution_status == "cancelled"
    assert state.status == "closed"
