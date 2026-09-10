"""Append-only message ledger for a persistent session."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# ``data:<mime>/<subtype>`` prefix; matches real inline-URI values, not prose
# that happens to start with "data:".
_DATA_URI = re.compile(r"data:[\w.+-]+/[\w.+-]+", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MessageLedger:
    """Persist the ordered, provider-shaped messages for one session."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._next_seq = self._read_next_seq()

    def _read_next_seq(self) -> int:
        if not self.path.exists():
            return 1
        next_seq = 1
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    next_seq = max(next_seq, int(json.loads(line)["seq"]) + 1)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
        return next_seq

    def append(self, message: Mapping[str, Any] | dict[str, Any]) -> int:
        seq = self._next_seq
        record = {"seq": seq, "timestamp": utc_now(), "message": _json_safe(dict(message))}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._next_seq += 1
        return seq

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def messages(self) -> list[dict[str, Any]]:
        return [record["message"] for record in self.records()]


def _json_safe(value: Any) -> Any:
    """Convert SDK/Pydantic values while redacting credential fields."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            result[str(key)] = (
                "[REDACTED]"
                if lowered
                in {
                    "api_key",
                    "authorization",
                    "x-api-key",
                    "token",
                    "secret",
                    "password",
                    "credential",
                    "credentials",
                }
                else _json_safe(item)
            )
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
    # Summarize only URI-shaped values (``data:<mime>,...``); short text that
    # merely starts with "data:" must survive verbatim in the ledger.
    if isinstance(value, str) and _DATA_URI.match(value):
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"[data value omitted; length={len(value)} sha256={digest}]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class EventLedger:
    """Append-only structured lifecycle events for one session."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._next_seq = self._read_next_seq()

    def _read_next_seq(self) -> int:
        if not self.path.exists():
            return 1
        next_seq = 1
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    next_seq = max(next_seq, int(json.loads(line)["seq"]) + 1)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
        return next_seq

    def append(self, event: Mapping[str, Any]) -> int:
        seq = self._next_seq
        record = {"seq": seq, "timestamp": utc_now(), **_json_safe(dict(event))}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._next_seq += 1
        return seq

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
