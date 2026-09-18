"""Validated configuration loading for the v0.0.1 harness."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class ConfigError(ValueError):
    """A user-actionable configuration failure."""

    def __init__(self, message: str, *, code: str = "config_error") -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key: SecretStr = Field(default=SecretStr(""), exclude=True, repr=False)
    timeout_s: float = Field(default=60, gt=0)
    max_tokens: int = Field(default=4096, gt=0)

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

    sessions_dir: Path = Path("sessions")
    system_prompt_file: Path = Path("prompts/system.md")
    checkpoint_file: Path = Path("checkpoints.sqlite3")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: ModelConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: str | Path = "config/default.toml") -> Self:
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
            ("sessions_dir", "sessions"),
            ("system_prompt_file", "prompts/system.md"),
            ("checkpoint_file", "checkpoints.sqlite3"),
        ):
            value = runtime.setdefault(key, default)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"runtime.{key} must be a non-empty path")

        runtime["sessions_dir"] = _resolve_path(base_dir, runtime["sessions_dir"])
        runtime["system_prompt_file"] = _resolve_path(base_dir, runtime["system_prompt_file"])
        checkpoint = Path(runtime["checkpoint_file"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(runtime["sessions_dir"]) / checkpoint
        runtime["checkpoint_file"] = str(checkpoint.resolve())
        raw["runtime"] = runtime

        try:
            config = cls.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"Invalid Config {config_path}: {exc}") from exc

        return config

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
            "sessions_dir": str(self.runtime.sessions_dir),
            "checkpoint_file": str(self.runtime.checkpoint_file),
            "system_prompt_file": str(self.runtime.system_prompt_file),
        }


def _resolve_path(base_dir: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    return str((candidate if candidate.is_absolute() else base_dir / candidate).resolve())
