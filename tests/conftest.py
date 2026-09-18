from __future__ import annotations

from pathlib import Path

import pytest

from rs_harness.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant. Workspace: {workdir}", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    sessions = tmp_path / "sessions"
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                'model = "test-model"',
                'api_key = "test-key"',
                "",
                "[runtime]",
                f'sessions_dir = "{sessions}"',
                f'system_prompt_file = "{prompt}"',
                'checkpoint_file = "checkpoints.sqlite3"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return Config.load(config_path)
