import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from rs_harness import cli
from rs_harness.config import ConfigError
from rs_harness.runtime.session import InteractionResult, SessionResult, SessionState


def _fake_session(
    session_id: str = "session-1",
    *,
    status: str = "running",
    history: list[dict[str, object]] | None = None,
):
    """Build the minimal session object consumed by ``_interactive_impl``."""

    state = SessionState(session_id=session_id, status=status)  # type: ignore[arg-type]

    class FakeSession:
        def __init__(self):
            self.state = state
            self.closed = False
            self.history = history or []

        def messages(self):
            return self.history

        def close(self):
            self.closed = True
            self.state.status = "closed"
            return SessionResult(
                session_id=self.state.session_id,
                status="closed",
                session_path="/tmp/session",
            )

    return FakeSession()


def test_format_session_history_keeps_user_and_assistant_text_only():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first question\nwith context"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "content": "large tool result"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "second question"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}, {"type": "image_ref"}],
        },
    ]

    assert cli._format_session_history(messages) == (
        "user> first question\nwith context\n"
        "user> second question\n"
        "assistant> answer"
    )


def test_ask_without_session_id_creates_one(monkeypatch, tmp_path: Path, capsys):
    config = object()
    created = SimpleNamespace(state=SimpleNamespace(session_id="generated"))
    seen = {}
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded, **kwargs: created)

    def fake_run(session_id, request, loaded, **kwargs):
        seen["session_id"] = session_id
        seen["run_kwargs"] = kwargs
        return InteractionResult(session_id=session_id, status="succeeded", answer="ok", session_status="running", turn=1)

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli._ask_impl("hello", None, None, tmp_path / "config.toml", False) == 0
    assert seen["session_id"] == "generated"
    assert seen["run_kwargs"]["tool_log"] is cli._cli_tool_log
    assert capsys.readouterr().out.strip() == "ok"


def test_ask_with_session_id_reuses_it(monkeypatch, tmp_path: Path):
    config = object()
    seen = {}
    monkeypatch.setattr(cli.Config, "load", lambda _: config)

    def fake_run(session_id, request, loaded, **kwargs):
        seen["session_id"] = session_id
        return InteractionResult(session_id=session_id, status="succeeded", answer="ok", session_status="running", turn=1)

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli._ask_impl("hello", None, "existing", tmp_path / "config.toml", False) == 0
    assert seen["session_id"] == "existing"


def test_ask_passes_explicit_workdir_to_session_and_loop(monkeypatch, tmp_path: Path):
    config = object()
    created = SimpleNamespace(state=SimpleNamespace(session_id="generated"))
    seen = {}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    def fake_create(loaded, **kwargs):
        seen["create"] = kwargs
        return created

    def fake_run(session_id, request, loaded, **kwargs):
        seen["run"] = kwargs
        return InteractionResult(session_id=session_id, status="succeeded", answer="ok", session_status="running", turn=1)

    monkeypatch.setattr(cli, "create_session", fake_create)
    monkeypatch.setattr(cli, "run", fake_run)

    assert cli._ask_impl("hello", None, None, tmp_path / "config.toml", False, workdir) == 0
    assert seen["create"]["workdir"] == workdir
    assert seen["run"]["workdir"] == workdir
    assert seen["run"]["tool_log"] is cli._cli_tool_log


def test_cli_tool_log_is_plain_without_tty(capsys):
    cli._cli_tool_log("bash", "output preview")

    assert capsys.readouterr().out == "> bash\noutput preview\n"


def test_list_sessions_impl_prints_summaries_as_json_array(monkeypatch, tmp_path: Path, capsys):
    config = object()
    states = [
        SessionState(session_id="s1", status="running", query="hello"),
        SessionState(session_id="s2", status="closed", turn=3, message_count=4),
    ]
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli.Session, "list", classmethod(lambda cls, loaded: states))

    assert cli._list_sessions_impl(tmp_path / "config.toml") == 0

    payload = json.loads(capsys.readouterr().out)
    assert [item["session_id"] for item in payload] == ["s1", "s2"]
    assert payload[0]["status"] == "running"
    assert payload[0]["query"] == "hello"
    assert payload[1]["turn"] == 3
    assert payload[1]["message_count"] == 4
    assert all("answer" not in item and "errors" not in item for item in payload)


def test_list_sessions_impl_reports_config_error(monkeypatch, tmp_path: Path, capsys):
    def fail_load(_):
        raise ConfigError("config_error", "boom")

    monkeypatch.setattr(cli.Config, "load", fail_load)

    assert cli._list_sessions_impl(tmp_path / "config.toml") == 1
    assert "config_error: boom" in capsys.readouterr().err


def test_interactive_creates_session_and_reuses_it_for_multiple_inputs(monkeypatch, tmp_path, capsys):
    config = object()
    session = _fake_session("generated")
    run_calls = []
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded: session)

    def fake_run(session_id, request, loaded, **kwargs):
        run_calls.append((session_id, request.query, kwargs))
        return InteractionResult(
            session_id=session_id,
            status="succeeded",
            answer=f"answer: {request.query}",
            session_status="running",
            turn=len(run_calls),
        )

    monkeypatch.setattr(cli, "run", fake_run)
    inputs = iter(["first question", "second question", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    assert cli._interactive_impl(tmp_path / "config.toml", None) == 0
    assert [(session_id, query) for session_id, query, _ in run_calls] == [
        ("generated", "first question"),
        ("generated", "second question"),
    ]
    assert all(kwargs["tool_log"] is cli._cli_tool_log for *_, kwargs in run_calls)
    output = capsys.readouterr().out
    assert "rsh session generated" in output
    assert "answer: first question" in output
    assert "answer: second question" in output
    assert "Session remains running: generated" in output
    assert "Session history:" not in output


def test_interactive_resume_does_not_create_new_session(monkeypatch, tmp_path, capsys):
    config = object()
    session = _fake_session(
        "existing",
        history=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "previous question"},
            {"role": "tool", "content": "tool result"},
            {"role": "assistant", "content": "previous answer"},
        ],
    )
    opened = []
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded: (_ for _ in ()).throw(AssertionError("must not create")))
    monkeypatch.setattr(cli.Session, "open", lambda loaded, session_id: opened.append(session_id) or session)
    inputs = iter(["/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    assert cli._interactive_impl(tmp_path / "config.toml", "existing") == 0
    assert opened == ["existing"]
    output = capsys.readouterr().out
    assert "Session history:\nuser> previous question\nassistant> previous answer" in output
    assert output.index("Session history:") < output.index("Type /help")
    assert "tool result" not in output
    assert "Resume with: uv run rsh --session-id existing" in output


def test_interactive_exit_alias_leaves_session_running(monkeypatch, tmp_path, capsys):
    config = object()
    session = _fake_session("existing")
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli.Session, "open", lambda loaded, session_id: session)
    monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

    assert cli._interactive_impl(tmp_path / "config.toml", "existing") == 0
    assert session.state.status == "running"
    output = capsys.readouterr().out
    assert "Session remains running: existing" in output
    assert "Resume with: uv run rsh --session-id existing" in output


def test_interactive_commands_help_status_unknown_and_blank_do_not_call_model(
    monkeypatch, tmp_path, capsys
):
    config = object()
    session = _fake_session("session-1")
    run_calls = []
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded: session)
    monkeypatch.setattr(cli.Session, "open", lambda loaded, session_id: session)
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: run_calls.append(args))
    inputs = iter(["", "  ", "/help", "/status", "/unknown", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    assert cli._interactive_impl(tmp_path / "config.toml", None) == 0
    output = capsys.readouterr()
    assert run_calls == []
    assert "/help  show commands" in output.out
    assert "/exit  leave session running" in output.out
    assert '"session_id": "session-1"' in output.out
    assert "Unknown command: /unknown" in output.err


def test_interactive_close_closes_session_and_exits(monkeypatch, tmp_path, capsys):
    config = object()
    session = _fake_session("session-1")
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded: session)
    monkeypatch.setattr("builtins.input", lambda _prompt: "/close")

    assert cli._interactive_impl(tmp_path / "config.toml", None) == 0
    assert session.closed is True
    assert '"status": "closed"' in capsys.readouterr().out


def test_interactive_eof_leaves_session_running(monkeypatch, tmp_path, capsys):
    config = object()
    session = _fake_session("session-1")
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded: session)

    def eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert cli._interactive_impl(tmp_path / "config.toml", None) == 0
    assert session.state.status == "running"
    assert "Session remains running: session-1" in capsys.readouterr().out


def test_configure_readline_applies_reference_bindings(monkeypatch):
    calls = []
    fake_readline = SimpleNamespace(parse_and_bind=calls.append)
    monkeypatch.setitem(sys.modules, "readline", fake_readline)

    cli._configure_readline()

    assert calls == [
        "set bind-tty-special-chars off",
        "set input-meta on",
        "set output-meta on",
        "set convert-meta off",
    ]


def test_configure_readline_ignores_missing_optional_module(monkeypatch):
    original_import = builtins.__import__

    def import_without_readline(name, *args, **kwargs):
        if name == "readline":
            raise ImportError("readline unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_readline)

    cli._configure_readline()
