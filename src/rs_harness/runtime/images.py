"""Attach images to model requests without ever persisting image bytes.

Only file signatures are examined here. Decoding pixels, georeferencing or any
other remote-sensing semantics stay behind the future MCP boundary.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MAX_IMAGE_BYTES",
    "ImageAttachment",
    "sniff_mime",
    "inspect_local_image",
    "extract_image_paths",
]

MAX_IMAGE_BYTES = 20 * 1024 * 1024

_SNIFF_LEN = 16

# Paths glued to surrounding text (e.g. "看这张图。/abs/path/a.jpg") need a
# substring scan; whitespace tokens additionally cover bare relative names.
_PATH_LIKE = re.compile(r"[~/]?[\w.\-]+(?:/[\w.\-]+)+")
_TRIM = "\"'`(),.;:!?[]{}。，；：！？）】》「」"


class ImageAttachment(BaseModel):
    """Evidence about one directly viewable local image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    mime: str
    bytes: int = Field(gt=0)
    sha256: str

    def ref_block(self) -> dict[str, Any]:
        """Ledger-safe block; image_url blocks are rebuilt at request time."""

        return {"type": "image_ref", **self.model_dump()}


def sniff_mime(head: bytes) -> str | None:
    """Identify vision-API-compatible image formats from leading bytes."""

    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def inspect_local_image(path: str | Path) -> tuple[ImageAttachment | None, str]:
    """Inspect one local file for direct vision input.

    Returns ``(attachment, "")`` on success or ``(None, reason)`` with a
    reason that is safe to embed in the user-facing limitation note.
    """

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return None, "file not found"
    size = file_path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        limit_mb = MAX_IMAGE_BYTES / 1024 / 1024
        return None, f"file is too large ({size / 1024 / 1024:.1f} MB > {limit_mb:.0f} MB)"
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        head = handle.read(_SNIFF_LEN)
        digest.update(head)
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    mime = sniff_mime(head)
    if mime is None:
        return None, "unsupported format; vision input supports JPEG, PNG, WebP, GIF"
    attachment = ImageAttachment(
        path=str(file_path.resolve()),
        mime=mime,
        bytes=size,
        sha256=digest.hexdigest(),
    )
    return attachment, ""


def extract_image_paths(query: str, *, base_dir: str | Path | None = None) -> list[str]:
    """Find existing local image files mentioned in a query.

    Candidates come from embedded path-like substrings and whitespace tokens;
    only files whose signature is a supported image format are returned, so
    prose mentioning ordinary files never triggers an attachment.
    """

    candidates = [match.group(0) for match in _PATH_LIKE.finditer(query)]
    candidates.extend(query.split())
    paths: list[str] = []
    for candidate in candidates:
        for variant in (candidate, candidate.strip(_TRIM)):
            resolved = _resolve_existing_image(variant, base_dir)
            if resolved is not None and resolved not in paths:
                paths.append(resolved)
    return paths


def _resolve_existing_image(candidate: str, base_dir: str | Path | None) -> str | None:
    text = candidate.strip()
    if not text:
        return None
    file_path = Path(text).expanduser()
    if base_dir is not None and not file_path.is_absolute():
        under_base = Path(base_dir) / file_path
        if under_base.is_file():
            file_path = under_base
    if not file_path.is_file():
        return None
    try:
        with file_path.open("rb") as handle:
            head = handle.read(_SNIFF_LEN)
    except OSError:
        return None
    if sniff_mime(head) is None:
        return None
    return str(file_path.resolve())
