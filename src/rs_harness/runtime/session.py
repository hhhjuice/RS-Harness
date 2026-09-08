"""Persistent session state and message-ledger management."""

from __future__ import annotations

import json
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from .trace import MessageLedger, _json_safe, utc_now
from .tools import normalize_workdir

SessionStatus = Literal["running", "closed", "failed"]
InteractionStatus = Literal["succeeded", "not_supported", "failed"]


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
    workdir: str | None = None
    turn: int = 0
    message_count: int = 0
    errors: list[ErrorInfo] = Field(default_factory=list)
    answer: str | None = None
    stop_reason: str | None = None
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None


class SessionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: SessionStatus
    answer: str | None = None
    errors: list[ErrorInfo] = Field(default_factory=list)
    session_path: str
    stop_reason: str | None = None


class InteractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: InteractionStatus
    answer: str | None = None
    errors: list[ErrorInfo] = Field(default_factory=list)
    session_status: SessionStatus
    turn: int


class Session:
    """Own one durable model context and its append-only message ledger."""

    def __init__(self, state: SessionState, path: Path):
        self.state = state
        self.path = path
        self.ledger = MessageLedger(path / "messages.jsonl")
        self._lock_path = path / "session.lock"

    @classmethod
    def create(
        cls,
        config: Config,
        *,
        query: str | None = None,
        image: str | None = None,
        workdir: str | Path | None = None,
    ) -> "Session":
        system_prompt = config.system_prompt()
        resolved_workdir = normalize_workdir(workdir)
        session_id = _session_id()
        path = config.runtime.sessions_dir / session_id
        path.mkdir(parents=True, exist_ok=False)
        session = cls(
            SessionState(
                session_id=session_id,
                mode="exploration" if query is None and image is not None else "instruction",
                query=query,
                image=image,
                workdir=resolved_workdir,
                started_at=utc_now(),
            ),
            path,
        )
        session.append({"role": "system", "content": system_prompt})
        session.save_state()
        return session

    @classmethod
    def open(cls, config: Config, session_id: str) -> "Session":
        if not session_id or Path(session_id).name != session_id:
            raise ValueError("invalid session_id")
        path = config.runtime.sessions_dir / session_id
        state_path = path / "session.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        state = SessionState.model_validate_json(state_path.read_text(encoding="utf-8"))
        return cls(state, path)

    @classmethod
    def list(cls, config: Config) -> list[SessionState]:
        """Load every persisted session state, oldest first.

        Entries without a readable ``session.json`` (stray files, partially
        written directories) are skipped; ``open`` remains the strict path.
        """

        sessions_dir = config.runtime.sessions_dir
        if not sessions_dir.is_dir():
            return []
        states: list[SessionState] = []
        for path in sorted(sessions_dir.iterdir()):
            state_path = path / "session.json"
            if not state_path.is_file():
                continue
            try:
                states.append(
                    SessionState.model_validate_json(state_path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
        return states

    @contextmanager
    def locked(self) -> Iterator["Session"]:
        import fcntl

        self._lock_path.touch(exist_ok=True)
        with self._lock_path.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self.reload()
            try:
                yield self
            finally:
                self.save_state()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def reload(self) -> None:
        self.state = SessionState.model_validate_json(
            (self.path / "session.json").read_text(encoding="utf-8")
        )
        self.ledger = MessageLedger(self.path / "messages.jsonl")

    def save_state(self) -> None:
        _atomic_json_write(self.path / "session.json", self.state.model_dump(mode="json"))

    def append_message(self, message: dict[str, Any]) -> int:
        with self.locked():
            return self.append(message)

    def append(self, message: dict[str, Any]) -> int:
        """Append one message; the caller must already hold the session lock."""
        if self.state.status != "running":
            raise ValueError("session_closed")
        seq = self.ledger.append(message)
        self.state.message_count = seq
        return seq

    def messages(self) -> list[dict[str, Any]]:
        return self.ledger.messages()

    def next_turn(self) -> int:
        if self.state.status != "running":
            raise ValueError("session_closed")
        self.state.turn += 1
        return self.state.turn

    def add_error(self, error: ErrorInfo) -> None:
        self.state.errors.append(error)

    def _last_assistant_answer_unlocked(self) -> str | None:
        messages = self.ledger.messages()
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            if message.get("tool_calls"):
                # A pending tool plan is not a final answer, and older answers
                # must not be reported as the result of this unfinished turn.
                return None
            content = message.get("content")
            if isinstance(content, str):
                return content.strip() or None
            if isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type", "text") == "text"
                    and isinstance(part.get("text"), str)
                ]
                answer = "\n".join(texts).strip()
                if answer:
                    return answer
                return None
            return None
        return None

    def close(
        self,
        status: Literal["closed", "failed"] = "closed",
        *,
        answer: str | None = None,
        errors: list[ErrorInfo] | None = None,
        stop_reason: str | None = None,
    ) -> SessionResult:
        with self.locked():
            if self.state.status in ("closed", "failed"):
                return SessionResult(
                    session_id=self.state.session_id,
                    status=self.state.status,
                    answer=self.state.answer,
                    errors=self.state.errors,
                    session_path=str(self.path),
                    stop_reason=self.state.stop_reason,
                )
            self.state.status = status
            self.state.errors = errors if errors is not None else self.state.errors
            resolved_answer = answer if answer is not None else self._last_assistant_answer_unlocked()
            self.state.answer = resolved_answer
            self.state.stop_reason = stop_reason
            self.state.finished_at = utc_now()
            result = SessionResult(
                session_id=self.state.session_id,
                status=status,
                answer=resolved_answer,
                errors=self.state.errors,
                session_path=str(self.path),
                stop_reason=stop_reason,
            )
            return result


def _session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(3)}"


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
