import json

import pytest
from typer.testing import CliRunner

from rs_harness.cli import app
from rs_harness.sessions.catalog import SessionCatalog


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
