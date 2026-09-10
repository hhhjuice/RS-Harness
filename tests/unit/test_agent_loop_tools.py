import base64
from pathlib import Path
from types import SimpleNamespace

import rs_harness.agent_loop as agent_loop_module
from rs_harness.agent_loop import InputRequest, create_session, run
from rs_harness.config import Config
from rs_harness.runtime.session import Session

JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-payload"


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
    prompt.write_text(
        "system\nWorking directory: {workdir}\n"
        "Only create, modify, or edit files inside this directory.",
        encoding="utf-8",
    )
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
    notices: list[str] = []

    result = run(
        session.state.session_id,
        InputRequest(query="do it"),
        config,
        openai_sdk_client=client,
        console=lambda message: notices.append(message),
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
    assert notices == [
        f"[HOOK] UserPromptSubmit: working in {workspace.resolve()}",
        "[HOOK] write_file(['a.txt', 'hello'])",
        "[HOOK] read_file(['a.txt'])",
        "[HOOK] Stop: session used 2 tool calls",
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
    notices: list[str] = []

    result = run(
        session.state.session_id,
        {"query": "recover"},
        config,
        openai_sdk_client=client,
        console=lambda message: notices.append(message),
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
    # An undecodable call has no arguments to preview, so only the valid
    # invocation is logged.
    assert "[HOOK] missing_tool([])" in notices
    assert not any("read_file(" in notice for notice in notices)


def test_console_previews_only_first_60_argument_characters(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient(
        [
            _response(calls=[_call("long", "glob", '{"pattern":"' + "x" * 100 + '"}')]),
            _response("done"),
        ]
    )
    session = create_session(config, workdir=workspace)
    notices: list[str] = []

    result = run(
        session.state.session_id,
        {"query": "find"},
        config,
        openai_sdk_client=client,
        console=lambda message: notices.append(message),
    )

    assert result.status == "succeeded"
    assert notices[1] == f"[HOOK] glob({str(['x' * 100])[:60]})"
    assert len(notices[1]) <= 60 + len("[HOOK] glob()")


def test_console_is_silent_by_default(tmp_path: Path, capsys):
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
    notices: list[str] = []

    result = run(
        session.state.session_id,
        {"query": "show env"},
        config,
        openai_sdk_client=client,
        console=lambda message: notices.append(message),
    )

    assert result.status == "succeeded"
    assert "[HOOK] bash(['env'])" in notices
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


def test_read_image_tool_attaches_image_after_tool_results(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "photo.jpg"
    image.write_bytes(JPEG)
    client = SequenceClient(
        [
            _response(
                calls=[
                    _call("image-1", "read_image", '{"path":"photo.jpg"}'),
                    _call("glob-1", "glob", '{"pattern":"*.jpg"}'),
                ]
            ),
            _response("I can see it"),
        ]
    )
    session = create_session(config, workdir=workspace)
    notices: list[str] = []

    result = run(
        session.state.session_id,
        {"query": "看看这个 jpg 图片有什么内容"},
        config,
        openai_sdk_client=client,
        console=lambda message: notices.append(message),
    )

    assert result.status == "succeeded"
    messages = client.calls[1]["messages"]
    assert [message["role"] for message in messages[-4:]] == [
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert messages[-2] == {
        "role": "tool",
        "tool_call_id": "glob-1",
        "content": "photo.jpg",
    }
    image_content = messages[-1]["content"]
    assert image_content[0]["type"] == "text"
    assert image_content[1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode("ascii")
        },
    }
    assert "[HOOK] read_image(['photo.jpg'])" in notices

    ledger = (session.path / "messages.jsonl").read_text(encoding="utf-8")
    assert "image_ref" in ledger
    assert "data:image/jpeg" not in ledger
    assert "jpeg-payload" not in ledger


def test_read_image_tool_history_reembeds_and_degrades_when_file_removed(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "photo.jpg"
    image.write_bytes(JPEG)
    client = SequenceClient(
        [
            _response(calls=[_call("image-1", "read_image", '{"path":"photo.jpg"}')]),
            _response("first"),
            _response("second"),
            _response("third"),
        ]
    )
    session = create_session(config, workdir=workspace)
    session_id = session.state.session_id

    run(session_id, {"query": "查看图片"}, config, openai_sdk_client=client)
    run(session_id, {"query": "补充说明"}, config, openai_sdk_client=client)

    history_images = [
        part
        for message in client.calls[2]["messages"]
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for part in message["content"]
    ]
    assert any(part.get("type") == "image_url" for part in history_images)

    image.unlink()
    run(session_id, {"query": "再次说明"}, config, openai_sdk_client=client)

    degraded = [
        part
        for message in client.calls[3]["messages"]
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for part in message["content"]
    ]
    assert {
        "type": "text",
        "text": f"Image previously attached at {image.resolve()} is no longer available.",
    } in degraded


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


def test_workdir_is_persisted_and_explicit_mismatch_is_rejected(tmp_path: Path):
    config = _config(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    session = create_session(config, workdir=first)
    client = SequenceClient([_response("ok")])

    result = run(
        session.state.session_id,
        {"query": "q"},
        config,
        openai_sdk_client=client,
        workdir=second,
    )

    assert result.status == "failed"
    assert result.errors[0].code == "workdir_mismatch"
    reopened = Session.open(config, session.state.session_id)
    assert reopened.state.workdir == str(first.resolve())
    assert len([message for message in reopened.messages() if message["role"] == "system"]) == 1


def test_workdir_is_reused_without_adding_system_messages(tmp_path: Path):
    config = _config(tmp_path)
    first = tmp_path / "first"
    first.mkdir()
    session = create_session(config, workdir=first)
    client = SequenceClient([_response("one"), _response("two"), _response("three")])
    run(session.state.session_id, {"query": "1"}, config, openai_sdk_client=client)
    run(session.state.session_id, {"query": "2"}, config, openai_sdk_client=client)
    run(session.state.session_id, {"query": "3"}, config, openai_sdk_client=client)
    messages = Session.open(config, session.state.session_id).messages()
    assert len([message for message in messages if message["role"] == "system"]) == 1


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
