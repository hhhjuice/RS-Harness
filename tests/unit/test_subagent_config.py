import pytest

from rs_harness.config import Config, ConfigError, SubagentConfig, subagents_digest


@pytest.mark.parametrize("update", [
    {"id": "Bad-Id"}, {"id": "worker\n"}, {"name": " "}, {"name": "line\nname"},
    {"name": "bad\x1bname"}, {"description": ""}, {"system_prompt_file": ""},
    {"tools": ["delegate_task"]}, {"tools": ["os.system"]}, {"tools": ["read_file", "read_file"]},
    {"tools": None}, {"persistence": "forever"}, {"model": "override"},
])
def test_invalid_definition_is_rejected(config, update):
    with pytest.raises(ValueError):
        SubagentConfig.model_validate({**config.subagents[0].model_dump(), **update})


@pytest.mark.parametrize("definitions", [[], None, "worker"])
def test_definitions_are_required_and_nonempty(config, definitions):
    with pytest.raises(ValueError):
        Config.model_validate({**config.model_dump(), "subagents": definitions})


def test_duplicate_id_and_missing_tools_fail(config):
    definition = config.subagents[0].model_dump()
    with pytest.raises(ValueError, match="unique"):
        Config.model_validate({**config.model_dump(), "subagents": [definition, definition]})
    del definition["tools"]
    with pytest.raises(ValueError, match="tools"):
        SubagentConfig.model_validate(definition)


@pytest.mark.parametrize("content", [b"  \n", b"\xff", None])
def test_unreadable_or_empty_prompt_fails_during_loading(config, tmp_path, content):
    path = tmp_path / "worker.md"
    if content is None:
        path.unlink()
    else:
        path.write_bytes(content)
    with pytest.raises(ConfigError, match="prompt"):
        Config.load(tmp_path / "config.toml")


def test_resolved_definitions_are_immutable_snapshots(config, tmp_path):
    assert config.subagents[0].system_prompt_file == tmp_path / "worker.md"
    original = config.resolve_subagents()
    (tmp_path / "worker.md").write_text("changed")
    assert original[0].system_prompt != config.resolve_subagents()[0].system_prompt
    assert subagents_digest(original) != subagents_digest(config.resolve_subagents())
    assert subagents_digest(original) == subagents_digest((original[0].model_copy(update={"name": "改名"}),))
    assert subagents_digest(original) == subagents_digest((original[0].model_copy(update={"tools": tuple(reversed(original[0].tools))}),))


def test_empty_tools_and_default_persistence(config):
    data = config.subagents[0].model_dump()
    data.pop("persistence")
    definition = SubagentConfig.model_validate({**data, "tools": []})
    assert definition.tools == () and definition.persistence == "per_invocation"
