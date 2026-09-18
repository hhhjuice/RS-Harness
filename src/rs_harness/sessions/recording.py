"""Append-only local trajectory recording through LangChain callbacks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from langchain_core.callbacks import AsyncCallbackHandler


class TrajectoryWriter:
    def __init__(self, path: Path, *, session_id: str, interaction_id: str):
        self.path = path
        self.session_id = session_id
        self.interaction_id = interaction_id
        self.failures: list[str] = []

    def append(
        self,
        event: str,
        payload: Any = None,
        *,
        run_id: str | UUID | None = None,
        parent_run_id: str | UUID | None = None,
    ) -> None:
        record = {
            "event_id": uuid4().hex,
            "interaction_id": self.interaction_id,
            "timestamp": _now(),
            "session_id": self.session_id,
            "event": event,
            "run_id": _stringify_id(run_id),
            "parent_run_id": _stringify_id(parent_run_id),
            "payload": _jsonable(payload),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
        except (OSError, TypeError, ValueError) as exc:
            self.failures.append(str(exc))


class TrajectoryRecorder(AsyncCallbackHandler):
    """Record model lifecycle events without changing agent execution."""

    raise_error = False

    def __init__(self, writer: TrajectoryWriter):
        super().__init__()
        self.writer = writer

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.writer.append(
            "model_start",
            {
                "model": serialized.get("id") or serialized.get("name"),
                "message_count": _message_count(messages),
            },
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.writer.append(
            "model_end",
            {"response": _jsonable(response)},
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.writer.append(
            "model_error",
            {"type": type(error).__name__, "message": str(error)},
            run_id=run_id,
            parent_run_id=parent_run_id,
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stringify_id(value: str | UUID | None) -> str | None:
    return None if value is None else str(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (UUID, Path)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    return str(value)


def _message_count(messages: list[Any]) -> int:
    if messages and isinstance(messages[0], (list, tuple)):
        return sum(len(batch) for batch in messages)
    return len(messages)
