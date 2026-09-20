import asyncio
import json

import pytest
from typer.testing import CliRunner

from rs_harness.cli import app
from rs_harness.sessions.catalog import SessionCatalog
from tests.images import write_test_image


def test_create_and_list_session(config, tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                'model = "test-model"',
                'api_key = "test-key"',
                "",
                "[runtime]",
                f'sessions_dir = "{config.runtime.sessions_dir}"',
                f'system_prompt_file = "{config.runtime.system_prompt_file}"',
                'checkpoint_file = "checkpoints.sqlite3"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    created = runner.invoke(app, ["create-session", "--config", str(config_path), "--json"])
    assert created.exit_code == 0, created.output
    assert json.loads(created.output)["session_id"]
    assert "thread_id" not in json.loads(created.output)
    listed = runner.invoke(app, ["list-session", "--config", str(config_path), "--json"])
    assert listed.exit_code == 0, listed.output
    assert "session_id" in listed.output
    assert all("thread_id" not in meta for meta in json.loads(listed.output))


@pytest.mark.parametrize("text", ["", "\n", "/quit\n", "/exit\n"])
def test_empty_or_exit_pipe_input_leaves_no_session(config, tmp_path, text):
    result = CliRunner().invoke(app, ["--config", str(tmp_path / "config.toml")], input=text)
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert "Warning:" not in result.output
    assert SessionCatalog.list(config) == []


@pytest.mark.parametrize("extra", [[], [""], ["   "]])
def test_empty_ask_creates_no_session(config, tmp_path, extra):
    result = CliRunner().invoke(app, ["ask", "--config", str(tmp_path / "config.toml"), *extra])
    assert result.exit_code == 0
    assert SessionCatalog.list(config) == []


def test_invalid_image_json_creates_no_session(config, tmp_path):
    result = CliRunner().invoke(app, ["ask", "--config", str(tmp_path / "config.toml"), "--image", "missing.png", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["type"] == "InputError"
    assert result.stderr == ""
    assert SessionCatalog.list(config) == []


def test_print_action_escapes_bash_control_characters(capsys):
    from rs_harness import cli

    cli._print_action({"name": "bash", "args": {"command": "printf '\x1b]52;c;secret\x07\n'"}})

    output = capsys.readouterr().err
    assert "\x1b" not in output
    assert "\\u001b" in output
    assert "\\u0007" in output
    assert "\\n" in output


def test_cli_image_only_and_resume_keep_workdir(config, tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from rs_harness import application, cli
    from tests.models import ScriptedModel

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    write_test_image(workdir / "sample.png")
    model = ScriptedModel([AIMessage(content="seen"), AIMessage(content="remembered")])

    async def ask_with_model(loaded_config, session_id, text):
        return await application.ask(loaded_config, session_id, text, model=model)

    monkeypatch.setattr(cli, "ask", ask_with_model)
    monkeypatch.chdir(tmp_path)
    options = ["--config", str(tmp_path / "config.toml"), "--json"]
    result = CliRunner().invoke(app, ["ask", *options, "--workdir", str(workdir), "--image", "sample.png", "--image", "https://example.com/sample.png"])
    assert result.exit_code == 0, result.output
    session_id = json.loads(result.stdout)["session_id"]
    blocks = model.inputs[0][-1].content
    assert [block["type"] for block in blocks] == ["image", "image"]
    continued = CliRunner().invoke(app, ["ask", "follow up", *options, "--session-id", session_id])
    assert continued.exit_code == 0, continued.output
    assert model.inputs[1][1].content == blocks
    failed = CliRunner().invoke(app, ["ask", *options, "--session-id", session_id, "--image", "missing.png"])
    assert failed.exit_code == 1
    assert model.call_count == 2
    changed_workdir = CliRunner().invoke(app, ["ask", "hello", *options, "--session-id", session_id, "--workdir", str(tmp_path)])
    assert changed_workdir.exit_code == 1
    assert "session_workdir_mismatch" in json.loads(changed_workdir.stdout)["error"]["message"]


@pytest.mark.parametrize("document", [
    "{", '{"id":{},"id":{}}', '{"id":{"decisions":[{"type":"approve","type":"reject"}]}}',
    'null', '[]', '{"old":{"decisions":[{"type":"edit"}]}}',
])
def test_resume_rejects_invalid_json_without_mutating_pending(config, tmp_path, document):
    from langchain_core.messages import AIMessage
    from rs_harness import application
    from tests.models import ScriptedModel

    session = application.create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="", tool_calls=[{
        "name": "bash", "args": {"command": "rm file.txt"}, "id": "delete", "type": "tool_call",
    }])])
    pending = asyncio.run(application.ask(config, session.session_id, "run", model=model))
    assert pending.status == "interrupted"
    before = session.trajectory_path.read_bytes()
    path = tmp_path / "decisions.json"
    path.write_text(document)
    result = CliRunner().invoke(app, [
        "resume", "--config", str(tmp_path / "config.toml"), "--session-id", session.session_id,
        "--decisions-file", str(path), "--json",
    ])
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["type"] == "invalid_approval_decisions"
    assert result.stderr == ""
    assert session.trajectory_path.read_bytes() == before
    assert asyncio.run(application.get_pending_approvals(config, session.session_id, model=model)).interrupts == pending.interrupts


@pytest.mark.parametrize("already_pending", [False, True])
def test_pipe_never_consumes_next_line_as_approval(config, tmp_path, monkeypatch, already_pending):
    from langchain_core.messages import AIMessage
    from rs_harness import application, cli
    from tests.models import ScriptedModel

    session = application.create_session(config, workdir=str(tmp_path))
    (tmp_path / "file.txt").write_text("keep")
    model = ScriptedModel([AIMessage(content="", tool_calls=[{
        "name": "bash", "args": {"command": "rm file.txt"}, "id": "delete", "type": "tool_call",
    }])])
    async def ask_with_model(config, session_id, text):
        return await application.ask(config, session_id, text, model=model)
    monkeypatch.setattr(cli, "ask", ask_with_model)
    if already_pending:
        asyncio.run(ask_with_model(config, session.session_id, "run"))
    result = CliRunner().invoke(app, [
        "--config", str(tmp_path / "config.toml"), "--session-id", session.session_id,
    ], input="run\nyes\n/close\n")
    assert result.exit_code == 2, result.output
    assert "Action 1:" not in result.output
    assert "rm file.txt" in result.stderr
    assert (tmp_path / "file.txt").read_text() == "keep"
    assert model.call_count == 1
    assert SessionCatalog.open(config, session.session_id).meta.status == "running"


def test_ask_json_reports_pending_gate(config, tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from rs_harness import application, cli
    from tests.models import ScriptedModel

    model = ScriptedModel([AIMessage(content="", tool_calls=[{
        "name": "bash", "args": {"command": "rm file.txt"}, "id": "delete", "type": "tool_call",
    }])])
    async def ask_with_model(config, session_id, text):
        return await application.ask(config, session_id, text, model=model)
    monkeypatch.setattr(cli, "ask", ask_with_model)
    options = ["--config", str(tmp_path / "config.toml"), "--json"]
    first = CliRunner().invoke(app, ["ask", "remove", *options, "--workdir", str(tmp_path)])
    assert first.exit_code == 2, first.output
    pending = json.loads(first.stdout)
    second = CliRunner().invoke(app, ["ask", "new", *options, "--session-id", pending["session_id"]])
    assert second.exit_code == 1, second.output
    result = json.loads(second.stdout)
    assert result["error"]["type"] == "session_pending_approval"
    assert result["interrupts"] == pending["interrupts"]
    assert result["interaction_id"] == pending["interaction_id"]
    assert first.stderr == second.stderr == ""


def test_resume_json_can_pause_again(config, tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from rs_harness import application, cli
    from tests.models import ScriptedModel

    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "bash", "args": {"command": "rm one.txt"}, "id": "one", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "bash", "args": {"command": "rm two.txt"}, "id": "two", "type": "tool_call"}]),
    ])
    session = application.create_session(config, workdir=str(tmp_path))
    first = asyncio.run(application.ask(config, session.session_id, "run", model=model))
    async def resume_with_model(config, session_id, decisions):
        return await application.resume(config, session_id, decisions, model=model)
    monkeypatch.setattr(cli, "resume", resume_with_model)
    decisions_file = tmp_path / "decisions.json"
    decisions_file.write_text(json.dumps({first.interrupts[0]["id"]: {"decisions": [{"type": "reject"}]}}))
    result = CliRunner().invoke(app, [
        "resume", "--config", str(tmp_path / "config.toml"), "--session-id", session.session_id,
        "--decisions-file", str(decisions_file), "--json",
    ])
    assert result.exit_code == 2, result.output
    assert result.stderr == ""
    pending = json.loads(result.stdout)
    assert pending["interaction_id"] == first.interaction_id
    assert pending["interrupts"][0]["id"] != first.interrupts[0]["id"]
    assert pending["interrupts"][0]["action_requests"][0]["args"] == {"command": "rm two.txt"}
