"""RSH session metadata and single-interaction locking."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..config import Config


class SessionError(ValueError):
    """A session lookup or lifecycle error."""


class SessionBusy(SessionError):
    """The session already has an active interaction."""


class SessionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: str = Field(pattern=r"^(running|closed)$")
    workdir: str
    model: str
    created_at: str
    updated_at: str
    closed_at: str | None = None

class Session:
    def __init__(self, meta: SessionMeta, path: Path):
        self.meta = meta
        self.path = path

    @property
    def session_id(self) -> str:
        return self.meta.session_id

    @property
    def trajectory_path(self) -> Path:
        return self.path / "trajectory.jsonl"

    @property
    def state_path(self) -> Path:
        return self.path / "session.json"

    def save(self) -> None:
        self.meta.updated_at = _now()
        _atomic_write(self.state_path, self.meta.model_dump_json(indent=2) + "\n")

    def close(self) -> None:
        if self.meta.status == "closed":
            return
        self.meta.status = "closed"
        self.meta.closed_at = _now()
        self.save()

    @contextmanager
    def interaction_lock(self) -> Iterator[None]:
        if self.meta.status != "running":
            raise SessionError("session_closed")
        lock_path = self.path / ".interaction.lock"
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+", encoding="utf-8") as handle:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SessionBusy("session_busy") from exc
            try:
                self.reload()
                if self.meta.status != "running":
                    raise SessionError("session_closed")
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def reload(self) -> None:
        try:
            self.meta = SessionMeta.model_validate_json(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SessionError(f"Unable to read session {self.session_id}: {exc}") from exc


class SessionCatalog:
    @staticmethod
    def create(config: Config, *, workdir: str | Path | None = None) -> Session:
        sessions_dir = config.runtime.sessions_dir
        sessions_dir.mkdir(parents=True, exist_ok=True)
        session_id = uuid4().hex
        path = sessions_dir / session_id
        path.mkdir()
        resolved_workdir = str(Path(workdir or Path.cwd()).expanduser().resolve())
        now = _now()
        session = Session(
            SessionMeta(
                session_id=session_id,
                status="running",
                workdir=resolved_workdir,
                model=config.model.model,
                created_at=now,
                updated_at=now,
            ),
            path,
        )
        session.save()
        session.trajectory_path.touch()
        return session

    @staticmethod
    def open(config: Config, session_id: str) -> Session:
        if not _valid_session_id(session_id):
            raise SessionError("invalid_session_id")
        path = config.runtime.sessions_dir / session_id
        state_path = path / "session.json"
        if not state_path.is_file():
            raise SessionError(f"session_not_found: {session_id}")
        try:
            meta = SessionMeta.model_validate_json(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SessionError(f"invalid_session: {session_id}") from exc
        if meta.session_id != session_id or path.name != session_id:
            raise SessionError("session_identity_mismatch")
        return Session(meta, path)

    @staticmethod
    def list(config: Config) -> list[Session]:
        sessions_dir = config.runtime.sessions_dir
        if not sessions_dir.is_dir():
            return []
        sessions: list[Session] = []
        for path in sorted(sessions_dir.iterdir()):
            state_path = path / "session.json"
            if not state_path.is_file():
                continue
            try:
                meta = SessionMeta.model_validate_json(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if meta.session_id != path.name:
                continue
            sessions.append(Session(meta, path))
        return sessions

    @staticmethod
    def close(config: Config, session_id: str) -> Session:
        session = SessionCatalog.open(config, session_id)
        with session.interaction_lock():
            session.close()
        return session


def _valid_session_id(value: str) -> bool:
    return bool(value and re.fullmatch(r"[0-9a-f]{32}", value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
