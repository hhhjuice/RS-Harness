from pathlib import Path

import pytest

from rs_harness.config import Config, ConfigError


def test_config_loads_dotenv_and_resolves_paths(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("system", encoding="utf-8")
    (tmp_path / ".env").write_text("TEST_RS_KEY=from-dotenv\n", encoding="utf-8")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[model]
provider = "openai"
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

    assert config.model.provider == "openai"
    assert config.api_key() == "from-dotenv"
    assert config.runtime.sessions_dir == (tmp_path / "sessions").resolve()
    assert config.system_prompt() == "system"


def test_missing_key_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_RS_KEY", raising=False)
    config = Config(
        model={"provider": "anthropic", "model": "claude", "api_key_env": "TEST_RS_KEY"}
    )
    with pytest.raises(ConfigError, match="Missing API key"):
        config.api_key()

