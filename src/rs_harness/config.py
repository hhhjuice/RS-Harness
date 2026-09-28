"""Validated configuration loading for the v0.0.3 harness."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import tomllib
import unicodedata
from pathlib import Path
from typing import Literal, Self, get_args

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class ConfigError(ValueError):
    """A user-actionable configuration failure."""

    def __init__(self, message: str, *, code: str = "config_error") -> None:
        self.code = code
        self.message = message
        super().__init__(message)


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
REASONING_EFFORTS = get_args(ReasoningEffort)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key: SecretStr = Field(default=SecretStr(""), exclude=True, repr=False)
    timeout_s: float = Field(default=60, gt=0)
    max_tokens: int = Field(default=4096, gt=0)
    reasoning_effort: ReasoningEffort | None = None

    @field_validator("model")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        return value or None


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sessions_dir: Path = Path("../.agents/sessions")
    system_prompt_file: Path = Path("../.agents/prompts/system.md")
    skills_dir: Path = Path("../.agents/skills")
    checkpoint_file: Path = Path("checkpoints.sqlite3")
    max_model_calls: int = Field(default=10, gt=0, strict=True)
    bash_timeout_s: float = Field(default=60, gt=0, strict=True, allow_inf_nan=False)
    max_tool_output_bytes: int = Field(default=65536, gt=0, strict=True)


SubagentTool = Literal["read_file", "write_file", "edit_file", "glob", "bash", "read_image", "write_todos", "load_skill"]
SUBAGENT_TOOLS = get_args(SubagentTool)


class SubagentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    description: str
    system_prompt_file: Path
    tools: tuple[SubagentTool, ...]
    persistence: Literal["per_invocation", "per_thread"] = "per_invocation"

    @field_validator("name", "description")
    @classmethod
    def readable_text(cls, value: str) -> str:
        if not value.strip() or any(unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in value):
            raise ValueError("must be non-empty text without control characters or line breaks")
        return value.strip()

    @field_validator("system_prompt_file", mode="before")
    @classmethod
    def non_blank_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("system_prompt_file must be a non-empty path")
        return value

    @field_validator("tools")
    @classmethod
    def unique_tools(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("tools must not contain duplicates")
        return value


class ResolvedSubagent(SubagentConfig):
    system_prompt: str


def subagents_digest(subagents: tuple[ResolvedSubagent, ...]) -> str:
    definitions = [
        {"id": item.id, "prompt": item.system_prompt, "tools": sorted(item.tools), "persistence": item.persistence}
        for item in sorted(subagents, key=lambda item: item.id)
    ]
    return hashlib.sha256(json.dumps(definitions, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: ModelConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    subagents: tuple[SubagentConfig, ...] = Field(min_length=1)

    @field_validator("subagents")
    @classmethod
    def unique_subagents(cls, value):
        if len({item.id for item in value}) != len(value):
            raise ValueError("subagent ids must be unique")
        return value

    @classmethod
    def load(cls, path: str | Path = ".agents/config.toml") -> Self:
        config_path = Path(path).expanduser().resolve()
        try:
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except FileNotFoundError as exc:
            raise ConfigError(f"Config file not found: {config_path}") from exc
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"Unable to read Config {config_path}: {exc}") from exc

        runtime = dict(raw.get("runtime") or {})
        base_dir = config_path.parent
        for key, default in (
            ("sessions_dir", "../.agents/sessions"),
            ("system_prompt_file", "../.agents/prompts/system.md"),
            ("skills_dir", "../.agents/skills"),
            ("checkpoint_file", "checkpoints.sqlite3"),
        ):
            value = runtime.setdefault(key, default)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"runtime.{key} must be a non-empty path")

        runtime["sessions_dir"] = _resolve_path(base_dir, runtime["sessions_dir"])
        runtime["system_prompt_file"] = _resolve_path(base_dir, runtime["system_prompt_file"])
        runtime["skills_dir"] = _resolve_path(base_dir, runtime["skills_dir"])
        checkpoint = Path(runtime["checkpoint_file"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(runtime["sessions_dir"]) / checkpoint
        runtime["checkpoint_file"] = str(checkpoint.resolve())
        raw["runtime"] = runtime
        for item in raw.get("subagents", []) if isinstance(raw.get("subagents"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("system_prompt_file"), str) and item["system_prompt_file"].strip():
                item["system_prompt_file"] = _resolve_path(base_dir, item["system_prompt_file"])

        try:
            config = cls.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"Invalid Config {config_path}: {exc}") from exc

        config.resolve_subagents()
        return config

    def resolve_subagents(self) -> tuple[ResolvedSubagent, ...]:
        resolved = []
        for item in self.subagents:
            try:
                prompt = item.system_prompt_file.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                raise ConfigError(f"Unable to read subagent prompt for {item.id}") from exc
            if not prompt:
                raise ConfigError(f"Subagent prompt must not be empty: {item.id}")
            resolved.append(ResolvedSubagent(**item.model_dump(), system_prompt=prompt))
        return tuple(resolved)

    def api_key(self) -> str:
        value = self.model.api_key.get_secret_value().strip()
        if not value:
            raise ConfigError(
                "Missing API key: set model.api_key in the TOML config",
                code="missing_api_key",
            )
        return value

    def system_prompt(self, workdir: str) -> str:
        try:
            prompt = self.runtime.system_prompt_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigError(f"Unable to read system prompt: {exc}") from exc
        prompt = prompt.strip()
        if not prompt:
            raise ConfigError("System prompt must not be empty")
        return prompt.replace("{workdir}", workdir)

    def summary(self) -> dict[str, object]:
        return {
            "model": self.model.model,
            "base_url": self.model.base_url,
            "timeout_s": self.model.timeout_s,
            "max_tokens": self.model.max_tokens,
            "reasoning_effort": self.model.reasoning_effort,
            "sessions_dir": str(self.runtime.sessions_dir),
            "checkpoint_file": str(self.runtime.checkpoint_file),
            "system_prompt_file": str(self.runtime.system_prompt_file),
            "skills_dir": str(self.runtime.skills_dir),
            "max_model_calls": self.runtime.max_model_calls,
            "bash_timeout_s": self.runtime.bash_timeout_s,
            "max_tool_output_bytes": self.runtime.max_tool_output_bytes,
            "subagents": [item.model_dump(mode="json") for item in self.subagents],
        }


def save_reasoning_effort(path: Path, effort: ReasoningEffort | None) -> None:
    config_path = path.expanduser().resolve()
    temporary: Path | None = None
    try:
        document = tomlkit.parse(config_path.read_text(encoding="utf-8"))
        model = document["model"]
        if effort is None:
            model.pop("reasoning_effort", None)
        else:
            model["reasoning_effort"] = effort

        updated = tomlkit.dumps(document)
        tomllib.loads(updated)
        mode = stat.S_IMODE(config_path.stat().st_mode)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=config_path.parent,
            prefix=f".{config_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
        temporary.chmod(mode)
        os.replace(temporary, config_path)
    except Exception as exc:
        raise ConfigError("Unable to save model.reasoning_effort") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _resolve_path(base_dir: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    return str((candidate if candidate.is_absolute() else base_dir / candidate).resolve())
