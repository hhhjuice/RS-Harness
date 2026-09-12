"""Append-only message ledger for a persistent session."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .redact import CREDENTIAL_KEYS, summarize_data_uri


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MessageLedger:
    """Persist the ordered, provider-shaped messages for one session.

    Records are cached in memory after one read; ``records``/``messages``
    return fresh list shells but share the inner dicts, which callers must
    treat as read-only.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = _read_jsonl(self.path)
        self._next_seq = _next_seq_after(self._records)

    def append(self, message: Mapping[str, Any] | dict[str, Any]) -> int:
        seq = self._next_seq
        record = {"seq": seq, "timestamp": utc_now(), "message": _json_safe(dict(message))}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records.append(record)
        self._next_seq += 1
        return seq

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def messages(self) -> list[dict[str, Any]]:
        return [record["message"] for record in self._records]


def _json_safe(value: Any) -> Any:
    """Convert SDK/Pydantic values while redacting credential fields."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            result[str(key)] = "[REDACTED]" if lowered in CREDENTIAL_KEYS else _json_safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return {str(key): _json_safe(item) for key, item in vars(value).items()}
    if isinstance(value, str):
        # Summarize only URI-shaped values (``data:<mime>,...``); short text
        # that merely starts with "data:" must survive verbatim in the ledger.
        summarized = summarize_data_uri(value)
        if summarized is not None:
            return summarized
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class EventLedger:
    """Append-only structured lifecycle events for one session.

    Records are cached in memory after one read; see ``MessageLedger``.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = _read_jsonl(self.path)
        self._next_seq = _next_seq_after(self._records)

    def append(self, event: Mapping[str, Any]) -> int:
        seq = self._next_seq
        record = {"seq": seq, "timestamp": utc_now(), **_json_safe(dict(event))}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records.append(record)
        self._next_seq += 1
        return seq

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load every well-formed ``seq`` record; corrupt lines are skipped."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                int(record["seq"])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            records.append(record)
    return records


def _next_seq_after(records: list[dict[str, Any]]) -> int:
    return max((int(record["seq"]) for record in records), default=0) + 1
