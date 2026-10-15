import pytest

from rs_harness.config import Config, ConfigError, MAIN_AGENT_TOOLS


@pytest.mark.parametrize("section,expected", [
    ("", MAIN_AGENT_TOOLS),
    ("[agent]\n", MAIN_AGENT_TOOLS),
    ("[agent]\ntools = []\n", ()),
    ('[agent]\ntools = ["read_file", "glob"]\n', ("read_file", "glob")),
])
def test_main_agent_tools_load_and_summary(config, tmp_path, section, expected):
    path = tmp_path / "config.toml"
    path.write_text(path.read_text() + "\n" + section)

    loaded = Config.load(path)

    assert loaded.agent.tools == expected
    assert loaded.summary()["agent"] == {"tools": list(expected), "skills": None}
    assert loaded.subagents == config.subagents


@pytest.mark.parametrize("value", [
    '"read_file"', "false", "1", "{}", "[true]", "[1]",
    '["unknown"]', '["*"]', '["READ_FILE"]', '["connect_mcp"]',
    '["mcp__docs__search"]', '["read_file", "read_file"]',
])
def test_main_agent_tools_reject_invalid_values(config, tmp_path, value):
    path = tmp_path / "config.toml"
    path.write_text(path.read_text() + f"\n[agent]\ntools = {value}\n")

    with pytest.raises(ConfigError, match=r"agent\.tools"):
        Config.load(path)


def test_main_agent_rejects_unknown_fields(config, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(path.read_text() + "\n[agent]\ntool = []\n")

    with pytest.raises(ConfigError, match=r"agent\.tool"):
        Config.load(path)


def test_disabling_delegation_still_validates_subagent_definitions(config, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(path.read_text() + "\n[agent]\ntools = []\n")
    config.subagents[0].system_prompt_file.unlink()

    with pytest.raises(ConfigError, match="Unable to read subagent prompt"):
        Config.load(path)
