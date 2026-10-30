from __future__ import annotations

from pathlib import Path

import pytest

from rs_harness.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant. Workspace: {workdir}", encoding="utf-8")
    worker_prompt = tmp_path / "worker.md"
    worker_prompt.write_text("Complete the delegated task and report evidence and unfinished work.", encoding="utf-8")
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
                'skills_dir = "skills"',
                'checkpoint_file = "checkpoints.sqlite3"',
                "",
                "[[subagents]]",
                'id = "worker"',
                'name = "通用任务助手"',
                'description = "Complete bounded tasks"',
                'system_prompt_file = "worker.md"',
                'tools = ["read_file", "write_file", "edit_file", "glob", "bash", "read_image", "write_todos", "load_skill"]',
                'persistence = "per_invocation"',
            ]
        ),
        encoding="utf-8",
    )
    return Config.load(config_path)


@pytest.fixture
def skills_config(config):
    target = config.runtime.skills_dir / "fixture-skill/SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text(
        "---\nname: fixture-skill\ndescription: Fixture skill for integration tests\n---\n"
        "FIXTURE-SKILL-BODY\nFIXTURE-INSTALL-EXAMPLE\n",
        encoding="utf-8",
    )
    return config
