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
