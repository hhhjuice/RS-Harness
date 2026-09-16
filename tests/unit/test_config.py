import pytest
from pydantic import ValidationError

from rs_harness.config import Config, ConfigError


def test_config_loads_dotenv_and_resolves_paths(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text(
        "system\nWorking directory: {workdir}", encoding="utf-8"
    )
    (tmp_path / ".env").write_text("TEST_RS_KEY=from-dotenv\n", encoding="utf-8")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[model]
model = "gpt-test"
api_key_env = "TEST_RS_KEY"
[runtime]
sessions_dir = "sessions"
system_prompt_file = "prompts/system.md"
dotenv_file = ".env"
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_RS_KEY", raising=False)
    config = Config.load(config_file)

    assert config.model.model == "gpt-test"
    assert config.api_key() == "from-dotenv"
    assert config.runtime.sessions_dir == (tmp_path / "sessions").resolve()
    assert config.system_prompt("/tmp/workdir") == (
        "system\nWorking directory: /tmp/workdir"
    )
    assert "provider" not in config.summary()


def test_missing_key_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_RS_KEY", raising=False)
    config = Config(
        model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"}
    )
    with pytest.raises(ConfigError, match="Missing API key"):
        config.api_key()


def test_system_prompt_requires_workdir_placeholder(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("system", encoding="utf-8")
    config = Config(
        model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"},
        runtime={"system_prompt_file": prompt},
    )

    with pytest.raises(ConfigError, match="must contain"):
        config.system_prompt("/tmp/workdir")


def test_system_prompt_rejects_empty_template(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("   ", encoding="utf-8")
    config = Config(
        model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"},
        runtime={"system_prompt_file": prompt},
    )

    with pytest.raises(ConfigError, match="must not be empty"):
        config.system_prompt("/tmp/workdir")


def test_subagent_prompt_defaults_render_and_validate_the_same_way(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "sub.md").write_text(
        "sub\nWorking directory: {workdir}", encoding="utf-8"
    )
    config = Config(
        model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"},
        runtime={"subagent_system_prompt_file": tmp_path / "prompts" / "sub.md"},
    )
    assert config.subagent_system_prompt("/tmp/workdir") == "sub\nWorking directory: /tmp/workdir"

    strict = Config(
        model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"},
        runtime={"subagent_system_prompt_file": tmp_path / "prompts" / "bad.md"},
    )
    (tmp_path / "prompts" / "bad.md").write_text("no placeholder", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain"):
        strict.subagent_system_prompt("/tmp/workdir")

    missing = Config(
        model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"},
        runtime={"subagent_system_prompt_file": tmp_path / "prompts" / "missing.md"},
    )
    with pytest.raises(ConfigError, match="Unable to read subagent system prompt"):
        missing.subagent_system_prompt("/tmp/workdir")


def test_subagent_prompt_file_defaults_relative_to_config_toml(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "subagent.md").write_text("sub {workdir}", encoding="utf-8")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[model]
model = "gpt-test"
api_key_env = "TEST_RS_KEY"
[runtime]
sessions_dir = "sessions"
""",
        encoding="utf-8",
    )
    config = Config.load(config_file)
    assert config.runtime.subagent_system_prompt_file == (tmp_path / "prompts" / "subagent.md").resolve()
    assert "subagent_system_prompt_file" in config.summary()


def test_provider_field_is_rejected():
    with pytest.raises(ValidationError):
        Config(
            model={
                "provider": "openai",
                "model": "gpt-test",
                "api_key_env": "TEST_RS_KEY",
            }
        )


def test_loop_defaults_to_twenty_turns():
    config = Config(model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"})
    assert config.loop.max_turns == 20


def test_loop_bash_timeout_defaults_to_120s_and_rejects_non_positive():
    config = Config(model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"})
    assert config.loop.bash_timeout_s == 120
    with pytest.raises(ValidationError):
        Config(
            model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"},
            loop={"bash_timeout_s": 0},
        )


def test_background_subagent_limit_defaults_and_is_positive():
    config = Config(model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"})
    assert config.runtime.max_background_subagents == 4
    assert config.summary()["max_background_subagents"] == 4
    with pytest.raises(ValidationError):
        Config(
            model={"model": "gpt-test", "api_key_env": "TEST_RS_KEY"},
            runtime={"max_background_subagents": 0},
        )


def test_background_subagent_limit_loads_from_toml(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("system {workdir}", encoding="utf-8")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[model]
model = "gpt-test"
api_key_env = "TEST_RS_KEY"
[runtime]
max_background_subagents = 2
""",
        encoding="utf-8",
    )
    config = Config.load(config_file)
    assert config.runtime.max_background_subagents == 2
