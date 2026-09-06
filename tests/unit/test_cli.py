from pathlib import Path
from types import SimpleNamespace

from rs_harness import cli
from rs_harness.runtime.session import InteractionResult, SessionResult, SessionState


def _fake_session(session_id: str = "session-1", *, status: str = "running"):
    """Build the minimal session object consumed by ``_interactive_impl``."""

    state = SessionState(session_id=session_id, status=status)  # type: ignore[arg-type]

    class FakeSession:
        def __init__(self):
            self.state = state
            self.closed = False

        def close(self):
            self.closed = True
            self.state.status = "closed"
            return SessionResult(
                session_id=self.state.session_id,
                status="closed",
                session_path="/tmp/session",
            )

    return FakeSession()


def test_ask_without_session_id_creates_one(monkeypatch, tmp_path: Path, capsys):
    config = object()
    created = SimpleNamespace(state=SimpleNamespace(session_id="generated"))
    seen = {}
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded, **kwargs: created)

    def fake_run(session_id, request, loaded):
        seen["session_id"] = session_id
        return InteractionResult(session_id=session_id, status="succeeded", answer="ok", session_status="running", turn=1)

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli._ask_impl("hello", None, None, tmp_path / "config.toml", False) == 0
    assert seen["session_id"] == "generated"
    assert capsys.readouterr().out.strip() == "ok"


def test_ask_with_session_id_reuses_it(monkeypatch, tmp_path: Path):
    config = object()
    seen = {}
    monkeypatch.setattr(cli.Config, "load", lambda _: config)

    def fake_run(session_id, request, loaded):
        seen["session_id"] = session_id
        return InteractionResult(session_id=session_id, status="succeeded", answer="ok", session_status="running", turn=1)

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli._ask_impl("hello", None, "existing", tmp_path / "config.toml", False) == 0
    assert seen["session_id"] == "existing"


def test_interactive_creates_session_and_reuses_it_for_multiple_inputs(monkeypatch, tmp_path, capsys):
    config = object()
    session = _fake_session("generated")
    run_calls = []
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded: session)
    monkeypatch.setattr(
        cli,
        "run",
        lambda session_id, request, loaded: (
            run_calls.append((session_id, request.query))
            or InteractionResult(
                session_id=session_id,
                status="succeeded",
                answer=f"answer: {request.query}",
                session_status="running",
                turn=len(run_calls),
            )
        ),
    )
    inputs = iter(["first question", "second question", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    assert cli._interactive_impl(tmp_path / "config.toml", None) == 0
    assert run_calls == [("generated", "first question"), ("generated", "second question")]
    output = capsys.readouterr().out
    assert "rsh session generated" in output
    assert "answer: first question" in output
    assert "answer: second question" in output
    assert "Session remains running: generated" in output


def test_interactive_resume_does_not_create_new_session(monkeypatch, tmp_path, capsys):
    config = object()
    session = _fake_session("existing")
    opened = []
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded: (_ for _ in ()).throw(AssertionError("must not create")))
    monkeypatch.setattr(cli.Session, "open", lambda loaded, session_id: opened.append(session_id) or session)
    inputs = iter(["/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    assert cli._interactive_impl(tmp_path / "config.toml", "existing") == 0
    assert opened == ["existing"]
    assert "Resume with: uv run rsh --session-id existing" in capsys.readouterr().out


def test_interactive_commands_help_status_unknown_and_blank_do_not_call_model(
    monkeypatch, tmp_path, capsys
):
    config = object()
    session = _fake_session("session-1")
    run_calls = []
    monkeypatch.setattr(cli.Config, "load", lambda _: config)
    monkeypatch.setattr(cli, "create_session", lambda loaded: session)
    monkeypatch.setattr(cli.Session, "open", lambda loaded, session_id: session)
    monkeypatch.setattr(cli, "run", lambda *args: run_calls.append(args))
    inputs = iter(["", "  ", "/help", "/status", "/unknown", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    assert cli._interactive_impl(tmp_path / "config.toml", None) == 0
    output = capsys.readouterr()
    assert run_calls == []
    assert "/help  show commands" in output.out
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
