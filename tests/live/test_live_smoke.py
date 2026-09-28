from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rs_harness import cli
from rs_harness.application import create_session
from rs_harness.config import Config, ConfigError


@pytest.mark.live
def test_real_model_interactive_multiturn(tmp_path, monkeypatch):
    if not Path(".agents/config.toml").is_file():
        pytest.skip("Create .agents/config.toml from .agents/example.toml to run live tests")
    config = Config.load()
    try:
        config.api_key()
    except ConfigError as exc:
        if exc.code != "missing_api_key":
            raise
        pytest.skip(exc.message)
    runtime = config.runtime.model_copy(update={
        "sessions_dir": tmp_path / "sessions",
        "checkpoint_file": tmp_path / "sessions" / "checkpoints.sqlite3",
    })
    config = config.model_copy(update={"runtime": runtime})
    session = create_session(config, workdir=tmp_path)
    monkeypatch.setattr(cli, "_load", lambda path: config)
    result = CliRunner().invoke(
        cli.app,
        ["--session-id", session.session_id],
        input=(
            "Remember the code RSH-49261. Reply with exactly: ok\n"
            "What code did I ask you to remember? Reply only with the code.\n"
            "Repeat that same code once more. Reply only with the code.\n"
            "/quit\n"
        ),
    )
    assert result.exit_code == 0, result.output
    assert "Error:" not in result.output, result.output
    events = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    completed = [event for event in events if event["event"] == "interaction_succeeded"]
    assert len(completed) == 3
    assert all("RSH-49261" in event["payload"]["answer"] for event in completed[1:])
