"""Attach images to model requests without ever persisting image bytes.

Only file signatures are examined here. Decoding pixels, georeferencing or any
other remote-sensing semantics stay behind the future MCP boundary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MAX_IMAGE_BYTES",
    "ImageAttachment",
    "sniff_mime",
    "inspect_local_image",
]

MAX_IMAGE_BYTES = 32 * 1024 * 1024

_SNIFF_LEN = 16

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
