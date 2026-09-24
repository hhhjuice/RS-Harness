import os

import pytest

from rs_harness.config import Config, ConfigError, ModelConfig, REASONING_EFFORTS, save_reasoning_effort


def test_config_resolves_paths_and_prompt(config):
    assert config.runtime.sessions_dir.is_dir() is False
    assert config.system_prompt("/tmp/work") == "You are a test assistant. Workspace: /tmp/work"
    assert config.runtime.checkpoint_file == config.runtime.sessions_dir / "checkpoints.sqlite3"
    assert config.runtime.max_model_calls == 10
    assert config.summary()["max_model_calls"] == 10
    assert config.summary()["bash_timeout_s"] == 60
    assert config.summary()["max_tool_output_bytes"] == 65536
    assert config.model.reasoning_effort is None
    assert config.summary()["reasoning_effort"] is None


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_reasoning_effort_accepts_openai_values(effort):
    model = ModelConfig.model_validate({"model": "test-model", "reasoning_effort": effort})
    assert model.reasoning_effort == effort


@pytest.mark.parametrize("effort", ["", "LOW", "auto", "off", 3, True])
def test_reasoning_effort_rejects_other_values(effort):
    with pytest.raises(ValueError):
        ModelConfig.model_validate({"model": "test-model", "reasoning_effort": effort})


def test_reasoning_effort_write_preserves_file_and_can_reset(config, tmp_path):
    path = tmp_path / "config.toml"
    original = path.read_text(encoding="utf-8").replace(
        "[model]\n", "# Existing configuration\n[model]\n# Model comment\n"
    )
    path.write_text(original, encoding="utf-8")
    path.chmod(0o640)

    save_reasoning_effort(path, "high")
    assert Config.load(path).model.reasoning_effort == "high"
    assert path.stat().st_mode & 0o777 == 0o640
    assert "# Existing configuration" in path.read_text(encoding="utf-8")
    assert "# Model comment" in path.read_text(encoding="utf-8")
    assert 'api_key = "test-key"' in path.read_text(encoding="utf-8")

    save_reasoning_effort(path, "low")
    assert Config.load(path).model.reasoning_effort == "low"
    assert path.read_text(encoding="utf-8").count("reasoning_effort =") == 1

    save_reasoning_effort(path, None)
    assert Config.load(path).model.reasoning_effort is None
    assert path.read_text(encoding="utf-8") == original
    assert path.stat().st_mode & 0o777 == 0o640


def test_reasoning_effort_save_failure_keeps_original(config, tmp_path, monkeypatch):
    from rs_harness import config as config_module

    path = tmp_path / "config.toml"
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("replacement refused")

    monkeypatch.setattr(config_module.os, "replace", fail_replace)
    with pytest.raises(ConfigError, match="Unable to save model.reasoning_effort"):
        save_reasoning_effort(path, "high")
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".config.toml.*.tmp")) == []


@pytest.mark.parametrize("field,value", [
    ("bash_timeout_s", 0), ("bash_timeout_s", -1), ("bash_timeout_s", True),
    ("bash_timeout_s", "60"), ("bash_timeout_s", float("inf")), ("bash_timeout_s", float("nan")),
    ("max_tool_output_bytes", 0), ("max_tool_output_bytes", -1),
    ("max_tool_output_bytes", True), ("max_tool_output_bytes", 1.5), ("max_tool_output_bytes", "10"),
])
def test_tool_limits_are_strict(config, field, value):
    from rs_harness.config import RuntimeConfig

    with pytest.raises(ValueError):
        RuntimeConfig.model_validate({**config.runtime.model_dump(), field: value})


@pytest.mark.parametrize("rendered", ["0", "-1", "true", "false", "1.0", "1.5", '"10"'])
def test_max_model_calls_must_be_a_positive_integer(tmp_path, rendered):
    prompt = tmp_path / "system.md"
    prompt.write_text("test", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                'model = "test-model"',
                'api_key = "test-key"',
                "",
                "[runtime]",
                f'max_model_calls = {rendered}',
                f'system_prompt_file = "{prompt}"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Invalid Config") as error:
        Config.load(config_path)
    assert "runtime.max_model_calls" in str(error.value)


def test_max_model_calls_can_be_configured(config, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                'model = "test-model"',
                'api_key = "test-key"',
                "",
                "[runtime]",
                "max_model_calls = 3",
                f'sessions_dir = "{config.runtime.sessions_dir}"',
                f'system_prompt_file = "{config.runtime.system_prompt_file}"',
            ]
        ),
        encoding="utf-8",
    )

    loaded = Config.load(config_path)
    assert loaded.runtime.max_model_calls == 3


def test_api_key_comes_from_toml(config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "different-environment-key")
    assert config.api_key() == "test-key"


@pytest.mark.parametrize("key_setting", ['', 'api_key = ""', 'api_key = "   "'])
def test_missing_api_key_is_actionable(tmp_path, monkeypatch, key_setting):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-environment-key")
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'[model]\nmodel = "test-model"\n{key_setting}\n')
    config = Config.load(config_path)
    with pytest.raises(ConfigError, match="model.api_key") as error:
        config.api_key()
    assert error.value.code == "missing_api_key"


def test_api_key_is_not_exported(config):
    secret = config.api_key()
    assert "api_key" not in config.model_dump()["model"]
    assert secret not in config.model_dump_json()
    assert secret not in repr(config)
    assert secret not in str(config.summary())


def test_invalid_config_does_not_echo_secret(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[model]\nmodel = "test"\napi_key = ["private-test-key"]\n')
    with pytest.raises(ConfigError) as error:
        Config.load(config_path)
    assert "private-test-key" not in str(error.value)


def test_dotenv_is_not_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("RSH_UNUSED_DOTENV_VALUE", raising=False)
    (tmp_path / ".env").write_text("RSH_UNUSED_DOTENV_VALUE=unused\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text('[model]\nmodel = "test"\napi_key = "toml-key"\n')
    config = Config.load(config_path)
    assert config.api_key() == "toml-key"
    assert "RSH_UNUSED_DOTENV_VALUE" not in os.environ
