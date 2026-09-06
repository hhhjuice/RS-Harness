"""Single-session state and result persistence."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from .trace import TraceWriter, _json_safe, utc_now

SessionStatus = Literal["running", "succeeded", "not_supported", "failed"]


class ErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: SessionStatus = "running"
    mode: Literal["instruction", "exploration"] = "instruction"
    query: str | None = None
    image: str | None = None
    turn: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)
    stop_reason: str | None = None
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None


class SessionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: Literal["succeeded", "not_supported", "failed"]
    answer: str | None = None
    errors: list[ErrorInfo] = Field(default_factory=list)
    session_path: str
    stop_reason: str | None = None


class Session:
    """Own the directory and files for exactly one valid invocation."""

    def __init__(self, state: SessionState, path: Path, trace: TraceWriter):
        self.state = state
        self.path = path
        self.trace = trace

    @classmethod
    def start(cls, query: str | None, image: str | None, config: Config) -> "Session":
        session_id = _session_id()
        path = config.runtime.sessions_dir / session_id
        path.mkdir(parents=True, exist_ok=False)
        state = SessionState(
            session_id=session_id,
            mode="instruction" if query is not None else "exploration",
            query=query,
            image=image,
            started_at=utc_now(),
        )
        session = cls(state, path, TraceWriter(path / "events.jsonl"))
        session.trace.write(
            "session_started",
            session_id=session_id,
            mode=state.mode,
            query=query,
            image=image,
            config=config.summary(),
        )
        session.save_state()
        return session

    def save_state(self) -> None:
        _atomic_json_write(self.path / "session.json", self.state.model_dump(mode="json"))

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        self.state.messages = _json_safe(messages)
        self.save_state()

    def finish(
        self,
        status: Literal["succeeded", "not_supported", "failed"],
        *,
        answer: str | None = None,
        errors: list[ErrorInfo] | None = None,
        stop_reason: str | None = None,
    ) -> SessionResult:
        self.state.status = status
        self.state.errors = errors or []
        self.state.stop_reason = stop_reason
        self.state.finished_at = utc_now()
        result = SessionResult(
            session_id=self.state.session_id,
            status=status,
            answer=answer,
            errors=errors or [],
            session_path=str(self.path),
            stop_reason=stop_reason,
        )
        self.save_state()
        self.trace.write(
            "session_finished",
            session_id=self.state.session_id,
            status=status,
            stop_reason=stop_reason,
            answer=answer,
            errors=[error.model_dump(mode="json") for error in result.errors],
        )
        _atomic_json_write(self.path / "result.json", result.model_dump(mode="json"))
        return result

    def error(self, error: ErrorInfo) -> SessionResult:
        self.trace.write(
            "session_error",
            session_id=self.state.session_id,
            code=error.code,
            message=error.message,
            details=error.details,
        )
        return self.finish("failed", errors=[error], stop_reason=error.code)


def _session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(3)}"


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
