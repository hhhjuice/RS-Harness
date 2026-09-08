from pathlib import Path
from types import SimpleNamespace

import rs_harness.agent_loop as agent_loop_module
from rs_harness.agent_loop import InputRequest, create_session, run
from rs_harness.config import Config
from rs_harness.runtime.session import Session


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def _config(tmp_path: Path, *, max_turns: int = 20) -> Config:
    prompt = tmp_path / "system.md"
    prompt.write_text("system", encoding="utf-8")
    return Config(
        model={"model": "test-model", "api_key_env": "UNUSED_RS_KEY"},
        loop={"max_turns": max_turns},
        runtime={
            "sessions_dir": tmp_path / "sessions",
            "system_prompt_file": prompt,
            "dotenv_file": tmp_path / "missing.env",
        },
    )


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


def test_tool_calls_are_executed_returned_by_id_and_loop_continues(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(
                calls=[
                    _call("write-1", "write_file", '{"path":"a.txt","content":"hello"}'),
                    _call("read-1", "read_file", '{"path":"a.txt"}'),
                ]
            ),
            _response("done"),
        ]
    )
    session = create_session(config, workdir=workspace)
    events: list[tuple[str, str]] = []

    result = run(
        session.state.session_id,
        InputRequest(query="do it"),
        config,
        openai_sdk_client=client,
        tool_log=lambda name, preview: events.append((name, preview)),
    )

    assert result.status == "succeeded"
    assert result.answer == "done"
    assert len(client.calls) == 2
    assert [message["role"] for message in client.calls[1]["messages"][-3:]] == [
        "assistant",
        "tool",
        "tool",
    ]
    assert [message["tool_call_id"] for message in client.calls[1]["messages"][-2:]] == [
        "write-1",
        "read-1",
    ]
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hello"
    assert events == [
        ("write_file", "Wrote 5 bytes to a.txt"),
        ("read_file", "hello"),
    ]


def test_tool_errors_and_invalid_json_are_observations_then_model_can_recover(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(
                calls=[
                    _call("bad-json", "read_file", "not-json"),
                    _call("unknown", "missing_tool", "{}"),
                ]
            ),
            _response("recovered"),
        ]
    )
    session = create_session(config, workdir=workspace)
    events: list[tuple[str, str]] = []

    result = run(
        session.state.session_id,
        {"query": "recover"},
        config,
        openai_sdk_client=client,
        tool_log=lambda name, preview: events.append((name, preview)),
    )

    assert result.status == "succeeded"
    tool_messages = [m for m in Session.open(config, session.state.session_id).messages() if m["role"] == "tool"]
    assert tool_messages[0]["tool_call_id"] == "bad-json"
    assert "Invalid JSON arguments" in tool_messages[0]["content"]
    assert tool_messages[1] == {
        "role": "tool",
        "tool_call_id": "unknown",
        "content": "Unknown tool: missing_tool",
    }
    assert [name for name, _ in events] == ["read_file", "missing_tool"]
    assert events[0][1].startswith("Error: Invalid JSON arguments")
    assert events[1][1] == "Unknown tool: missing_tool"


def test_tool_log_contains_only_first_200_result_characters(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(agent_loop_module, "dispatch_tool", lambda *args, **kwargs: "x" * 250)
    client = SequenceClient(
        [
            _response(calls=[_call("long", "glob", '{"pattern":"*"}')]),
            _response("done"),
        ]
    )
    session = create_session(config, workdir=workspace)
    events: list[tuple[str, str]] = []

    result = run(
        session.state.session_id,
        {"query": "find"},
        config,
        openai_sdk_client=client,
        tool_log=lambda name, preview: events.append((name, preview)),
    )

    assert result.status == "succeeded"
    assert events == [("glob", "x" * 200)]


def test_tool_log_is_silent_by_default(tmp_path: Path, capsys):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(calls=[_call("glob-1", "glob", '{"pattern":"*"}')]),
            _response("done"),
        ]
    )
    session = create_session(config, workdir=workspace)

    result = run(session.state.session_id, {"query": "find"}, config, openai_sdk_client=client)

    assert result.status == "succeeded"
    assert capsys.readouterr().out == ""


def test_tool_output_secrets_are_redacted_in_log_ledger_and_model_request(
    tmp_path: Path, monkeypatch
):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("UNUSED_RS_KEY", "sk-live-test-secret")
    monkeypatch.setattr(
        agent_loop_module,
        "dispatch_tool",
        lambda *args, **kwargs: "OPENAI_API_KEY=sk-live-test-secret",
    )
    client = SequenceClient(
        [
            _response(calls=[_call("leak", "bash", '{"command":"env"}')]),
            _response("done"),
        ]
    )
    session = create_session(config, workdir=workspace)
    events: list[tuple[str, str]] = []

    result = run(
        session.state.session_id,
        {"query": "show env"},
        config,
        openai_sdk_client=client,
        tool_log=lambda name, preview: events.append((name, preview)),
    )

    assert result.status == "succeeded"
    assert events == [("bash", "OPENAI_API_KEY=[REDACTED]")]
    sent_tool_messages = [
        message
        for message in client.calls[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert sent_tool_messages[0]["content"] == "OPENAI_API_KEY=[REDACTED]"
    ledger = (session.path / "messages.jsonl").read_text(encoding="utf-8")
    assert "sk-live-test-secret" not in ledger
    assert "OPENAI_API_KEY=[REDACTED]" in ledger


def test_glob_tool_is_available_to_the_model_loop(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "one.txt").write_text("one", encoding="utf-8")
    client = SequenceClient(
        [
            _response(calls=[_call("glob-1", "glob", '{"pattern":"*.txt"}')]),
            _response("found"),
        ]
    )
    session = create_session(config, workdir=workspace)

    result = run(session.state.session_id, {"query": "find it"}, config, openai_sdk_client=client)

    assert result.status == "succeeded"
    assert {
        "role": "tool",
        "tool_call_id": "glob-1",
        "content": "one.txt",
    } in Session.open(config, session.state.session_id).messages()


def test_budget_executes_last_tool_batch_before_returning(tmp_path: Path):
    config = _config(tmp_path, max_turns=1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient([_response(calls=[_call("one", "write_file", '{"path":"x","content":"y"}')])])
    session = create_session(config, workdir=workspace)

    result = run(session.state.session_id, {"query": "write"}, config, openai_sdk_client=client)

    assert result.status == "failed"
    assert result.errors[0].code == "budget_exceeded"
    messages = Session.open(config, session.state.session_id).messages()
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "one"
    assert not any(m.get("role") == "tool" and not m.get("tool_call_id") for m in messages)


def test_invalid_call_id_is_rejected_before_assistant_is_persisted(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient([_response(calls=[_call("", "read_file", '{"path":"x"}')])])
    session = create_session(config, workdir=workspace)

    result = run(session.state.session_id, {"query": "read"}, config, openai_sdk_client=client)

    assert result.status == "failed"
    assert result.errors[0].code == "model_response_error"
    assert not any(m.get("role") == "assistant" for m in Session.open(config, session.state.session_id).messages())


def test_malformed_tool_calls_container_is_model_response_error(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient([_response(calls="not-a-list")])
    session = create_session(config, workdir=workspace)

    result = run(session.state.session_id, {"query": "read"}, config, openai_sdk_client=client)

    assert result.status == "failed"
    assert result.errors[0].code == "model_response_error"


def test_duplicate_call_ids_are_rejected_before_any_tool_runs(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(
                calls=[
                    _call("same", "write_file", '{"path":"x","content":"one"}'),
                    _call("same", "write_file", '{"path":"y","content":"two"}'),
                ]
            )
        ]
    )
    session = create_session(config, workdir=workspace)

    result = run(session.state.session_id, {"query": "write"}, config, openai_sdk_client=client)

    assert result.status == "failed"
    assert result.errors[0].code == "model_response_error"
    assert not (workspace / "x").exists()
    assert not (workspace / "y").exists()


def test_workdir_is_persisted_and_explicit_run_override_is_used(tmp_path: Path):
    config = _config(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    session = create_session(config, workdir=first)
    client = SequenceClient([_response("ok")])

    result = run(session.state.session_id, {"query": "q"}, config, openai_sdk_client=client, workdir=second)

    assert result.status == "succeeded"
    reopened = Session.open(config, session.state.session_id)
    assert reopened.state.workdir == str(second.resolve())
    assert any(
        message.get("role") == "system" and str(second.resolve()) in message.get("content", "")
        for message in reopened.messages()
    )


def test_workdir_change_back_adds_current_context_note(tmp_path: Path):
    config = _config(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    session = create_session(config, workdir=first)
    client = SequenceClient([_response("one"), _response("two"), _response("three")])
    run(session.state.session_id, {"query": "1"}, config, openai_sdk_client=client)
    run(session.state.session_id, {"query": "2"}, config, openai_sdk_client=client, workdir=second)
    run(session.state.session_id, {"query": "3"}, config, openai_sdk_client=client, workdir=first)
    notes = [m for m in Session.open(config, session.state.session_id).messages() if m.get("role") == "system" and "tool context" in m.get("content", "")]
    assert [str(first.resolve()) in n["content"] for n in notes] == [True, False, True]


def test_unmatched_historical_tool_call_is_not_replayed(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    session.append_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "old", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        }
    )
    client = SequenceClient([_response("must not run")])

    result = run(session.state.session_id, {"query": "continue"}, config, openai_sdk_client=client)

    assert result.status == "failed"
    assert result.errors[0].code == "unmatched_tool_call"
    assert client.calls == []


def test_interleaved_historical_message_before_tool_result_is_rejected(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    session.append_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "old", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        }
    )
    session.append_message({"role": "assistant", "content": "interleaved"})
    session.append_message({"role": "tool", "tool_call_id": "old", "content": "late"})

    result = run(session.state.session_id, {"query": "continue"}, config, openai_sdk_client=SequenceClient([]))

    assert result.status == "failed"
    assert result.errors[0].code == "unmatched_tool_call"
