from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rs_harness.config import Config, ConfigError, SubagentConfig, subagents_digest
from rs_harness.skills import SkillLoader


FIND_SKILLS = Path(__file__).resolve().parents[2] / ".agents/skills/find-skills/SKILL.md"


def manifest(root, directory, content):
    path = root / directory / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_real_skill_discovery_and_immutable_snapshot(tmp_path):
    original = FIND_SKILLS.read_text()
    path = manifest(tmp_path, "find-skills", original)
    (tmp_path / ".DS_Store").write_text("ignored")
    (tmp_path / "empty").mkdir()
    manifest(tmp_path / "nested", "ignored", "# Nested")
    loader = SkillLoader(tmp_path)
    assert list(loader.skills) == ["find-skills"]
    skill = loader.load("find-skills")
    assert skill.path == path and skill.content == original
    assert "Helps users discover and install agent skills" in loader.catalog()
    assert "## What is the Skills CLI?" not in loader.catalog()
    with pytest.raises(TypeError):
        loader.skills["new"] = skill
    with pytest.raises(FrozenInstanceError):
        skill.content = "changed"
    path.unlink()
    assert loader.load("find-skills").content == original
    assert SkillLoader(tmp_path).catalog() == "none"


@pytest.mark.parametrize("content,name,description", [
    ("\n# Fallback title\nBody", "fallback", "Fallback title"),
    ("---\nname: explicit\n---\n## Body description\n", "explicit", "Body description"),
    ("---\nname: ' '\ndescription: null\n---\n### Fallback\n", "fallback", "Fallback"),
    ("---\nname: [wrong]\ndescription: 42\n---\n# Fallback\n", "fallback", "Fallback"),
    ("---\nname: explicit\ndescription: |\n  first line\n  second   line\n---\nBody", "explicit", "first line second line"),
    ("---\nname: explicit\ndescription: |\n\n  ## Heading\n  detail\n---\nBody", "explicit", "Heading detail"),
    ("---\nname: explicit\ndescription: '  # Header  '\nallowed-tools: [bash]\n---\n", "explicit", "Header"),
    ("---\r\nname: explicit\r\ndescription: Original\r\n---\r\nBody\r\n", "explicit", "Original"),
])
def test_frontmatter_and_fallbacks_preserve_full_text(tmp_path, content, name, description):
    manifest(tmp_path, "fallback", content)
    skill = SkillLoader(tmp_path).load(name)
    assert skill.description == description
    assert skill.content == content


@pytest.mark.parametrize("content,reason", [
    ("---\nname: [\n---\nDO-NOT-LOG", "invalid YAML"),
    ("---\n- name\n---\nDO-NOT-LOG", "must be a mapping"),
    ("---\nname: value\nDO-NOT-LOG", "missing closing"),
    ("---\nname: bad/name\ndescription: short\n---\nDO-NOT-LOG", "path separators"),
    ('---\nname: "bad\\\\name"\ndescription: short\n---\nDO-NOT-LOG', "path separators"),
    ('---\nname: "bad\\nname"\ndescription: short\n---\nDO-NOT-LOG', "control characters"),
    ('---\nname: "bad\\u001bname"\ndescription: short\n---\nDO-NOT-LOG', "control characters"),
    ("---\nname: empty-description\n---\n", "description must not be empty"),
])
def test_invalid_manifest_logs_path_and_safe_reason(tmp_path, caplog, capsys, content, reason):
    path = manifest(tmp_path, "invalid", content)
    manifest(tmp_path, "valid", "# Still usable")
    loader = SkillLoader(tmp_path)
    assert list(loader.skills) == ["valid"]
    assert str(path) in caplog.text and reason in caplog.text
    assert "DO-NOT-LOG" not in caplog.text
    assert capsys.readouterr().out == ""


def test_invalid_encoding_and_unreadable_manifest_are_diagnosed(tmp_path, caplog, monkeypatch):
    bad = manifest(tmp_path, "encoding", "placeholder")
    bad.write_bytes(b"\xffBODY-SECRET")
    denied = manifest(tmp_path, "denied", "# No access")
    original = Path.open

    def open_path(self, *args, **kwargs):
        if self == denied:
            raise PermissionError("BODY-SECRET")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    assert SkillLoader(tmp_path).catalog() == "none"
    assert "PermissionError" in caplog.text and "UnicodeDecodeError" in caplog.text
    assert str(bad) in caplog.text and str(denied) in caplog.text
    assert "BODY-SECRET" not in caplog.text


def test_stable_directory_order_and_duplicate_names(tmp_path):
    second = manifest(tmp_path, "z", "---\nname: shared\ndescription: second\n---\n")
    first = manifest(tmp_path, "a", "---\nname: first\ndescription: first\n---\n")
    assert list(SkillLoader(tmp_path).skills) == ["first", "shared"]
    first.write_text("---\nname: shared\ndescription: first\n---\n")
    with pytest.raises(ConfigError, match="Duplicate skill") as error:
        SkillLoader(tmp_path)
    assert str(first) in str(error.value) and str(second) in str(error.value)


def test_missing_empty_and_invalid_root(tmp_path, monkeypatch):
    assert SkillLoader(tmp_path / "missing").catalog() == "none"
    assert SkillLoader(tmp_path).catalog() == "none"
    file = tmp_path / "file"
    file.touch()
    with pytest.raises(ConfigError, match="skills directory"):
        SkillLoader(file)

    def denied(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "iterdir", denied)
    with pytest.raises(ConfigError, match="skills directory"):
        SkillLoader(tmp_path)


def test_symlink_containment_and_name_only_lookup(tmp_path, caplog):
    root = tmp_path / "skills"
    internal = manifest(root, "internal", "# Internal")
    outside = manifest(tmp_path, "outside", "# External secret")
    link = root / "file-link" / "SKILL.md"
    link.parent.mkdir()
    link.symlink_to(outside)
    (root / "directory-link").symlink_to(outside.parent, target_is_directory=True)
    inside_link = root / "inside-link" / "SKILL.md"
    inside_link.parent.mkdir()
    inside_link.symlink_to(internal)
    loader = SkillLoader(root)
    assert set(loader.skills) == {"internal", "inside-link"}
    assert "outside skills directory" in caplog.text
    for name in ("../outside", str(outside), "internal/SKILL.md", "unknown", "", " "):
        with pytest.raises(KeyError):
            loader.load(name)


def test_config_resolves_skills_against_toml_not_cwd(config, tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    manifest(tmp_path, "skills/example", "# Example")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    loaded = Config.load(path)
    assert loaded.runtime.skills_dir == tmp_path / "skills"
    assert loaded.summary()["skills_dir"] == str(tmp_path / "skills")
    path.write_text(path.read_text().replace('skills_dir = "skills"\n', ""))
    assert Config.load(path).runtime.skills_dir == tmp_path.parent / ".agents/skills"
    path.write_text(path.read_text().replace("[runtime]", '[runtime]\nskills_dir = "~/rsh-test-skills"'))
    assert Config.load(path).runtime.skills_dir == Path.home() / "rsh-test-skills"


@pytest.mark.parametrize("value", ['""', '"  "', "false", "42"])
def test_invalid_skills_path_is_config_error(config, tmp_path, value):
    path = tmp_path / "config.toml"
    path.write_text(path.read_text().replace('skills_dir = "skills"', f"skills_dir = {value}"))
    with pytest.raises(ConfigError, match="runtime.skills_dir"):
        Config.load(path)


def test_skill_tool_is_explicit_and_affects_subagent_digest(config):
    definition = SubagentConfig.model_validate({**config.subagents[0].model_dump(), "tools": ["load_skill"]})
    assert definition.tools == ("load_skill",)
    original = config.resolve_subagents()
    changed = (original[0].model_copy(update={"tools": ()}),)
    assert subagents_digest(original) != subagents_digest(changed)
