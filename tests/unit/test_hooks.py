import json
from pathlib import Path
from types import SimpleNamespace

from rs_harness.agent_loop import InputRequest, create_session, run
from rs_harness.config import Config
from rs_harness.runtime.hooks import HookContext, HookDecision, HookRegistry, PermissionRequest, default_hooks
from rs_harness.runtime.session import Session
import rs_harness.agent_loop as agent_loop_module


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return next(self.responses)


def _config(tmp_path: Path, *, max_turns: int = 20) -> Config:
    prompt = tmp_path / "system.md"
    prompt.write_text("system\nWorking directory: {workdir}", encoding="utf-8")
    return Config(model={"model": "test", "api_key_env": "UNUSED_RS_KEY"}, loop={"max_turns": max_turns}, runtime={"sessions_dir": tmp_path / "sessions", "system_prompt_file": prompt, "dotenv_file": tmp_path / "missing.env"})


def _response(content=None, calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or [], role="assistant"))])


def _call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def test_registry_short_circuits_pretool_and_records_order():
    registry = HookRegistry()
    seen = []
    registry.register("PreToolUse", lambda ctx: seen.append("first") or HookDecision(False, "blocked"))
    registry.register("PreToolUse", lambda ctx: seen.append("second"))
    decision = registry.trigger(HookContext("PreToolUse", None, "s", 1, "/tmp", tool_name="bash"))
    assert seen == ["first"]
    assert decision.reason == "blocked"


def test_default_post_tool_hook_order_and_audit_sink():
    events = []
    registry = default_hooks(event_sink=events.append)
    assert [name for name, _ in registry.callbacks("PreToolUse")] == ["check_permission", "log_hook"]
    assert [name for name, _ in registry.callbacks("PostToolUse")] == [
        "tool_error_hook", "large_output_hook"
    ]
    context = HookContext("PostToolUse", None, "s", 1, "/tmp", tool_name="glob", output="ok")
    registry.trigger(context)
    assert [event["hook"] for event in events] == [
        "tool_error_hook", "large_output_hook"
    ]


def test_log_hook_previews_tool_invocations_before_execution():
    notices = []
    registry = default_hooks(console=lambda message: notices.append(message))
    context = HookContext(
        "PreToolUse", None, "s", 1, "/tmp", tool_name="bash",
        arguments={"command": "ls -la"},
    )
    decision = registry.trigger(context)

    assert decision is None
    assert notices == ["[HOOK] bash(['ls -la'])"]


def test_log_hook_skipped_when_permission_is_denied():
    notices = []
    registry = default_hooks(
        permission_callback=lambda request: False,
        console=lambda message: notices.append(message),
    )
    context = HookContext(
        "PreToolUse", None, "s", 1, "/tmp", tool_name="bash",
        arguments={"command": "rm file"},
    )
    decision = registry.trigger(context)

    assert decision is not None and not decision.allowed
    assert notices == []


def test_permission_denial_and_all_hook_events_are_persisted(tmp_path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response(calls=[_call("danger", "bash", '{"command":"rm file"}')]), _response("done")])
    approvals = []
    result = run(session.state.session_id, InputRequest(query="do it"), config, openai_sdk_client=client, permission_callback=lambda req: approvals.append(req) or False)
    assert result.status == "succeeded"
    assert approvals and isinstance(approvals[0], PermissionRequest)
    messages = Session.open(config, session.state.session_id).messages()
    assert messages[-2]["role"] == "tool" and "Permission denied" in messages[-2]["content"]
    events = [json.loads(line) for line in (session.path / "events.jsonl").read_text().splitlines()]
    event_names = [event["event"] for event in events]
    assert event_names[0:2] == ["UserPromptSubmit", "PreToolUse"]
    assert event_names[-1] == "Stop"
    assert event_names.count("PostToolUse") == 2
    assert any(event["status"] == "denied" for event in events)


def test_hard_deny_never_calls_approval_callback(tmp_path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response(calls=[_call("danger", "bash", '{"command":"sudo reboot"}')]), _response("done")])
    approvals = []
    result = run(session.state.session_id, {"query": "do it"}, config, openai_sdk_client=client, permission_callback=lambda req: approvals.append(req) or True)
    assert result.status == "succeeded"
    assert approvals == []


def test_permission_callback_allows_workspace_escape_for_one_call(tmp_path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response(calls=[_call("write", "write_file", json.dumps({"path": str(outside), "content": "ok"}))]), _response("done")])
    result = run(session.state.session_id, {"query": "write"}, config, openai_sdk_client=client, permission_callback=lambda req: True)
    assert result.status == "succeeded"
    assert outside.read_text() == "ok"


def test_permission_callback_allows_read_image_workspace_escape(tmp_path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"\xff\xd8\xff\xe0" + b"jpeg-payload")
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response(calls=[_call("img", "read_image", json.dumps({"path": str(outside)}))]), _response("done")])
    result = run(session.state.session_id, {"query": "看图"}, config, openai_sdk_client=client, permission_callback=lambda req: True)
    assert result.status == "succeeded"
    messages = Session.open(config, session.state.session_id).messages()
    observation = next(message for message in messages if message.get("role") == "tool")
    assert observation["content"].startswith("Image loaded:")


def test_event_arguments_are_truncated(tmp_path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    content = "x" * 1000
    client = SequenceClient([
        _response(calls=[_call("w", "write_file", json.dumps({"path": "big.txt", "content": content}))]),
        _response("done"),
    ])
    result = run(session.state.session_id, {"query": "write"}, config, openai_sdk_client=client)

    assert result.status == "succeeded"
    raw = (session.path / "events.jsonl").read_text(encoding="utf-8")
    assert content not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    pre_tool = next(
        event for event in events
        if event["event"] == "PreToolUse" and event["tool_name"] == "write_file"
    )
    assert len(pre_tool["arguments"]["content"]) == 200
    assert pre_tool["arguments"]["path"] == "big.txt"


def test_stop_hook_runs_when_budget_exceeded(tmp_path):
    config = _config(tmp_path, max_turns=1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response(calls=[_call("x", "glob", '{"pattern":"*"}')])])
    result = run(session.state.session_id, {"query": "find"}, config, openai_sdk_client=client)
    assert result.status == "failed"
    events = [json.loads(line) for line in (session.path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "Stop"


def test_hook_event_redacts_secret_output(tmp_path, monkeypatch):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("UNUSED_RS_KEY", "secret-value")
    monkeypatch.setattr(agent_loop_module, "dispatch_tool", lambda *args, **kwargs: "token=secret-value")
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response(calls=[_call("x", "glob", '{"pattern":"*"}')]), _response("done")])
    run(session.state.session_id, {"query": "find"}, config, openai_sdk_client=client)
    raw = (session.path / "events.jsonl").read_text()
    assert "secret-value" not in raw


def test_tool_exception_is_normalized_by_post_tool_hook(tmp_path, monkeypatch):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(agent_loop_module, "dispatch_tool", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response(calls=[_call("x", "glob", '{"pattern":"*"}')]), _response("recovered")])
    result = run(session.state.session_id, {"query": "find"}, config, openai_sdk_client=client)
    assert result.status == "succeeded"
    message = Session.open(config, session.state.session_id).messages()[-2]
    assert message["content"] == "Error: Tool execution failed: boom"


def test_tool_error_hook_redacts_credentials_and_data_uri(tmp_path):
    registry = default_hooks(redactor=lambda value: value)
    context = HookContext(
        "PostToolUse", None, "s", 1, "/tmp", tool_name="bash",
        error=RuntimeError("token=secret data:image/png;base64,AAA"),
    )
    registry.trigger(context)
    assert context.output == "Error: Tool execution failed: token=[REDACTED] [DATA_URI_REDACTED]"


def test_custom_post_tool_hook_can_replace_tool_error_output(tmp_path, monkeypatch):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(agent_loop_module, "dispatch_tool", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    custom = HookRegistry()
    custom.register("PostToolUse", lambda ctx: setattr(ctx, "output", "custom error") if ctx.error else None, name="custom_error")
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response(calls=[_call("x", "glob", '{"pattern":"*"}')]), _response("recovered")])
    result = run(session.state.session_id, {"query": "find"}, config, openai_sdk_client=client, hooks=custom)
    assert result.status == "succeeded"
    message = Session.open(config, session.state.session_id).messages()[-2]
    assert message["content"] == "custom error"


def test_post_tool_hooks_receive_redacted_output(tmp_path, monkeypatch):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("UNUSED_RS_KEY", "secret-value")
    monkeypatch.setattr(agent_loop_module, "dispatch_tool", lambda *args, **kwargs: "token=secret-value")
    seen = []
    custom = HookRegistry()
    custom.register("PostToolUse", lambda ctx: seen.append(ctx.output), name="capture")
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response(calls=[_call("x", "glob", '{"pattern":"*"}')]), _response("done")])
    run(session.state.session_id, {"query": "find"}, config, openai_sdk_client=client, hooks=custom)
    assert seen == ["token=[REDACTED]"]


def test_console_receives_workdir_invocation_and_summary(tmp_path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    notices = []
    client = SequenceClient([
        _response(calls=[_call("x", "glob", '{"pattern":"*.txt"}')]),
        _response("done"),
    ])

    result = run(
        session.state.session_id, {"query": "find"}, config,
        openai_sdk_client=client, console=lambda message: notices.append(message),
    )

    assert result.status == "succeeded"
    assert notices == [
        f"[HOOK] UserPromptSubmit: working in {workspace.resolve()}",
        "[HOOK] glob(['*.txt'])",
        "[HOOK] Stop: session used 1 tool calls",
    ]


def test_stop_hook_force_continues_the_loop(tmp_path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    custom = HookRegistry()
    forced = {"done": False}

    def force_once(ctx):
        if not forced["done"]:
            forced["done"] = True
            return "Continue: also report the total size."
        return None

    custom.register("Stop", force_once, name="force_once")
    client = SequenceClient([_response("first answer"), _response("final answer")])

    result = run(
        session.state.session_id, {"query": "report"}, config,
        openai_sdk_client=client, hooks=custom,
    )

    assert result.status == "succeeded"
    assert result.answer == "final answer"
    messages = Session.open(config, session.state.session_id).messages()
    assert messages[-3]["role"] == "assistant" and messages[-3]["content"] == "first answer"
    assert messages[-2] == {"role": "user", "content": "Continue: also report the total size."}
    assert messages[-1]["role"] == "assistant" and messages[-1]["content"] == "final answer"


def test_stop_force_continue_is_audited_and_budget_still_applies(tmp_path):
    config = _config(tmp_path, max_turns=2)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    custom = HookRegistry()
    custom.register("Stop", lambda ctx: "Continue forever.", name="always_force")
    client = SequenceClient([_response("one"), _response("two")])

    result = run(
        session.state.session_id, {"query": "report"}, config,
        openai_sdk_client=client, hooks=custom,
    )

    assert result.status == "failed"
    assert result.errors[0].code == "budget_exceeded"
    events = [json.loads(line) for line in (session.path / "events.jsonl").read_text().splitlines()]
    statuses = [event["status"] for event in events if event["event"] == "Stop"]
    # Each Stop trigger records one row per hook: summary_hook, then the
    # forcing custom hook. Two in-loop forces plus one failed exit.
    assert statuses == [
        "succeeded", "force_continue",
        "succeeded", "force_continue",
        "failed", "force_continue",
    ]


def test_stop_hook_fires_exactly_once_on_success(tmp_path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(config, workdir=workspace)
    client = SequenceClient([_response("done")])

    result = run(session.state.session_id, {"query": "hi"}, config, openai_sdk_client=client)

    assert result.status == "succeeded"
    events = [json.loads(line) for line in (session.path / "events.jsonl").read_text().splitlines()]
    stop_events = [event for event in events if event["event"] == "Stop"]
    assert len(stop_events) == 1
    assert stop_events[0]["status"] == "succeeded"
