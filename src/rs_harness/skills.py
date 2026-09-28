"""Immutable snapshots of the configured local skill library."""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from .config import ConfigError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    content: str


def _parse_skill(path: Path, content: str, directory_name: str) -> Skill:
    lines = content.splitlines(keepends=True)
    metadata = {}
    body = content
    if lines and lines[0].rstrip("\r\n") == "---":
        end = next((index for index in range(1, len(lines)) if lines[index].rstrip("\r\n") == "---"), None)
        if end is None:
            raise ValueError("missing closing frontmatter delimiter")
        try:
            metadata = yaml.safe_load("".join(lines[1:end]))
        except yaml.YAMLError as exc:
            raise ValueError("invalid YAML frontmatter") from exc
        if not isinstance(metadata, dict):
            raise ValueError("frontmatter must be a mapping")
        body = "".join(lines[end + 1:])

    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        name = directory_name
    if any(char in "/\\" or unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in name):
        raise ValueError("skill name must not contain path separators or control characters")
    name = name.strip()
    if not name:
        raise ValueError("skill name must not be empty")

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        description = body.strip().splitlines()[0] if body.strip() else ""
    description = " ".join(description.strip().lstrip("# ").split())
    if not description:
        raise ValueError("skill description must not be empty")
    return Skill(name, description, path, content)


class SkillLoader:
    """Scan once per Agent build; loading by name never accesses the filesystem."""

    def __init__(self, root: Path):
        self.skills = MappingProxyType(self._scan(root.expanduser().resolve()))

    @staticmethod
    def _scan(root: Path) -> dict[str, Skill]:
        try:
            directories = sorted(root.iterdir())
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise ConfigError(f"Unable to read skills directory: {root}") from exc

        skills = {}
        for directory in directories:
            manifest = directory / "SKILL.md"
            try:
                if not directory.is_dir():
                    continue
                path = manifest.resolve()
                if not path.is_relative_to(root):
                    raise ValueError("manifest resolves outside skills directory")
                if not path.is_file():
                    continue
                with path.open(encoding="utf-8", newline="") as handle:
                    skill = _parse_skill(path, handle.read(), directory.name)
            except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, UnicodeError) else type(exc).__name__
                logger.warning("Skipping skill %s: %s", manifest, reason)
                continue
            if skill.name in skills:
                raise ConfigError(f"Duplicate skill {skill.name!r}: {skills[skill.name].path} and {skill.path}")
            skills[skill.name] = skill
        return skills

    def catalog(self) -> str:
        return "\n".join(f"- {skill.name}: {skill.description}" for skill in self.skills.values()) or "none"

    def load(self, name: str) -> Skill:
        return self.skills[name]
