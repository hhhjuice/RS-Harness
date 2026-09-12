"""Single text and value scrubbing shared by ledger, audit, and errors."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

Redactor = Callable[[str], str]

# Dict keys whose whole value is dropped wherever a structure is persisted.
CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "x-api-key",
        "token",
        "secret",
        "password",
        "credential",
        "credentials",
    }
)

# The loose form matches inline URIs anywhere in free text; the anchored form
# decides when a whole string is itself a data value worth summarizing.
_DATA_URI = re.compile(
    r"data:[\w.+-]+/[\w.+-]+(?:;[\w=.+-]+)*,[^\s\"'<>]+", re.IGNORECASE
)
_DATA_URI_PREFIX = re.compile(r"data:[\w.+-]+/[\w.+-]+", re.IGNORECASE)
_CREDENTIAL = re.compile(
    r"(\b(?:api[_-]?key|authorization|x-api-key|token|secret|password|credentials?)\b\s*[=:]\s*)"
    r"(\[REDACTED\]|\"[^\"]*\"|'[^']*'|Bearer\s+[^\s,;}\]]+|Basic\s+[^\s,;}\]]+|[^\s,;}\]]+)",
    re.IGNORECASE,
)


def identity(value: str) -> str:
    return value


def redact_text(text: str, redactor: Redactor = identity) -> str:
    """Scrub one text: configured secret, inline data URIs, credential pairs."""

    text = redactor(text)
    text = _DATA_URI.sub("[DATA_URI_REDACTED]", text)
    return _CREDENTIAL.sub(r"\1[REDACTED]", text)


def redact_value(value: Any, redactor: Redactor = identity) -> Any:
    """Recursively scrub a structure for the hook audit ledger."""

    if isinstance(value, str):
        return redact_text(value, redactor)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": redact_text(str(value), redactor)}
    if isinstance(value, dict):
        return {str(key): redact_value(item, redactor) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(item, redactor) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(repr(value), redactor)


def secret_redactor(secret: str) -> Redactor:
    """Replace one configured secret value wherever it appears in text."""

    if not secret:
        return identity

    def replace(value: str) -> str:
        return value.replace(secret, "[REDACTED]")

    return replace


def summarize_data_uri(value: str) -> str | None:
    """Summarize a whole-string data URI to a digest, or None if not one."""

    if _DATA_URI_PREFIX.match(value):
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"[data value omitted; length={len(value)} sha256={digest}]"
    return None


__all__ = [
    "Redactor",
    "CREDENTIAL_KEYS",
    "identity",
    "redact_text",
    "redact_value",
    "secret_redactor",
    "summarize_data_uri",
]
