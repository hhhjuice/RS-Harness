"""Configuration loading and validation for RS-Harness.

Configuration is deliberately file-first. Only the environment variable named
by ``model.api_key_env`` supplies a secret; other environment variables never
override values in the TOML file.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigError(ValueError):
    """A user-actionable configuration failure.

    ``code`` is safe to expose in a session result. The exception never stores
    the actual API key.
    """

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str = Field(min_length=1)
    timeout_s: float = Field(default=60, gt=0)
    max_tokens: int = Field(default=4096, gt=0)

    @field_validator("model", "api_key_env")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("api_key_env")
    @classmethod
    def valid_env_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must be a valid environment variable name")
        return value

    @field_validator("base_url", mode="before")
    @classmethod
    def empty_url_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        return value or None


class LoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_turns: int = Field(default=20, gt=0)
    bash_timeout_s: float = Field(default=120, gt=0)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sessions_dir: Path = Path("sessions")
    system_prompt_file: Path = Path("prompts/system.md")
    dotenv_file: Path = Path(".env")


class Config(BaseModel):
    """Validated, immutable runtime configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: str | Path = "config/default.toml") -> Self:
        """Load TOML and the configured dotenv file.

        Relative paths in ``[runtime]`` are resolved relative to the TOML file,
        which makes a Config snapshot reproducible regardless of the caller's
        current working directory.
        """

        config_path = Path(path).expanduser()
        try:
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except FileNotFoundError as exc:
            raise ConfigError("config_error", f"Config file not found: {config_path}") from exc
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError("config_error", f"Unable to read Config {config_path}: {exc}") from exc

        if not isinstance(raw, dict):  # pragma: no cover - tomllib always returns dict
            raise ConfigError("config_error", "Config root must be a TOML table")

        runtime = dict(raw.get("runtime") or {})
        base_dir = config_path.resolve().parent
        runtime.setdefault("sessions_dir", "sessions")
        runtime.setdefault("system_prompt_file", "prompts/system.md")
        runtime.setdefault("dotenv_file", ".env")
        for key in ("sessions_dir", "system_prompt_file", "dotenv_file"):
            if not isinstance(runtime[key], str) or not runtime[key].strip():
                raise ConfigError("config_error", f"runtime.{key} must be a non-empty path")
            candidate = Path(runtime[key]).expanduser()
            runtime[key] = str((candidate if candidate.is_absolute() else base_dir / candidate).resolve())
        raw["runtime"] = runtime

        try:
            config = cls.model_validate(raw)
        except Exception as exc:
            raise ConfigError("config_error", f"Invalid Config {config_path}: {exc}") from exc

        _load_dotenv(config.runtime.dotenv_file)
        return config

    def api_key(self) -> str:
        """Return the configured API key without ever logging it."""

        value = os.getenv(self.model.api_key_env, "").strip()
        if not value:
            raise ConfigError(
                "config_error",
                f"Missing API key environment variable: {self.model.api_key_env}",
            )
        return value

    def system_prompt(self, workdir: str) -> str:
        prompt = self._read_prompt(self.runtime.system_prompt_file, "system prompt")
        if not prompt.strip():
            raise ConfigError("config_error", "System prompt must not be empty")
        if "{workdir}" not in prompt:
            raise ConfigError(
                "config_error", "System prompt must contain the {workdir} placeholder"
            )
        return prompt.replace("{workdir}", workdir)

    @staticmethod
    def _read_prompt(path: Path, label: str) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigError("config_error", f"Unable to read {label} {path}: {exc}") from exc

    def summary(self) -> dict[str, object]:
        """Return a safe Config summary suitable for trace events."""

        return {
            "model": self.model.model,
            "base_url": self.model.base_url,
            "api_key_env": self.model.api_key_env,
            "timeout_s": self.model.timeout_s,
            "max_tokens": self.model.max_tokens,
            "max_turns": self.loop.max_turns,
            "sessions_dir": str(self.runtime.sessions_dir),
            "system_prompt_file": str(self.runtime.system_prompt_file),
            "dotenv_file": str(self.runtime.dotenv_file),
        }


def _load_dotenv(path: Path) -> None:
    """Load dotenv values without overriding already-exported variables."""

    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # dependency is declared, but fail clearly if omitted
        raise ConfigError("config_error", "python-dotenv is required to load .env files") from exc

    # A missing .env is valid when deployment injects the same variable into the
    # process. api_key() performs the definitive presence check.
    load_dotenv(dotenv_path=path, override=False)
