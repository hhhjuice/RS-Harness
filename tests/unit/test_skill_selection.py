from dataclasses import FrozenInstanceError

import pytest

from rs_harness.config import Config, ConfigError, subagents_digest
from rs_harness.skills import SkillCatalog, SkillLoader
from tests.unit.test_skills import manifest


def configure_skills(path, owner, value):
    content = path.read_text()
    if value is not None:
        if owner == "main":
            content += f"\n[agent]\nskills = {value}\n"
        else:
            content = content.replace("[[subagents]]\n", f"[[subagents]]\nskills = {value}\n")
    path.write_text(content)


@pytest.mark.parametrize("owner", ["main", "worker"])
@pytest.mark.parametrize("value,expected", [
    (None, None), ("[]", ()), ('["alpha"]', ("alpha",)),
    ('["beta", "alpha"]', ("beta", "alpha")),
])
def test_skill_selection_config_and_summary(config, tmp_path, owner, value, expected):
    path = tmp_path / "config.toml"
    configure_skills(path, owner, value)

    loaded = Config.load(path)

    definition = loaded.agent if owner == "main" else loaded.subagents[0]
    summary = loaded.summary()["agent"] if owner == "main" else loaded.summary()["subagents"][0]
    assert definition.skills == expected
    assert summary["skills"] == (None if expected is None else list(expected))
    assert definition.tools == (config.agent.tools if owner == "main" else config.subagents[0].tools)


@pytest.mark.parametrize("owner", ["main", "worker"])
@pytest.mark.parametrize("value", [
    '"alpha"', "true", "42", "{}", "[1]", "[true]",
    '[""]', '["  "]', '["*"]', '[" * "]', '["alpha", "alpha"]',
])
def test_invalid_skill_selections_fail_during_config_loading(config, tmp_path, owner, value):
    path = tmp_path / "config.toml"
    configure_skills(path, owner, value)
    field = r"agent\.skills" if owner == "main" else r"subagents\.0\.skills"

    with pytest.raises(ConfigError, match=field):
        Config.load(path)


def test_skill_views_share_objects_and_remain_independent_and_immutable(tmp_path):
    manifest(tmp_path, "directory", "---\nname: canonical\ndescription: Canonical description\n---\nBODY")
    manifest(tmp_path, "fallback", "# Fallback description")
    loader = SkillLoader(tmp_path)

    selected = loader.select(("canonical",), owner="Main agent")
    other = loader.select(("fallback",), owner="Subagent worker")

    assert selected.load("canonical") is loader.load("canonical")
    assert other.load("fallback") is loader.load("fallback")
    assert set(loader.skills) == {"canonical", "fallback"}
    assert loader.select(None, owner="Main agent") is loader
    assert loader.select((), owner="Main agent").catalog() == "none"
    assert "fallback" not in selected.catalog() and "canonical" not in other.catalog()
    with pytest.raises(KeyError):
        selected.load("fallback")
    with pytest.raises(TypeError):
        selected.skills["fallback"] = other.load("fallback")
    with pytest.raises(FrozenInstanceError):
        selected.skills = {}
    with pytest.raises(ConfigError, match="Subagent worker.*directory"):
        loader.select(("directory",), owner="Subagent worker")
    source = dict(loader.skills)
    view = SkillCatalog(source)
    source.clear()
    assert set(view.skills) == {"canonical", "fallback"}


def test_subagent_digest_includes_skills_but_ignores_selection_order(config):
    original = config.resolve_subagents()[0]
    selected = original.model_copy(update={"skills": ("alpha", "beta")})
    reversed_selection = original.model_copy(update={"skills": ("beta", "alpha")})
    restricted = original.model_copy(update={"skills": ("alpha",)})
    empty = original.model_copy(update={"skills": ()})

    assert subagents_digest((selected,)) == subagents_digest((reversed_selection,))
    assert len({subagents_digest((item,)) for item in (original, selected, restricted, empty)}) == 4
