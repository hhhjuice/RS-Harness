"""Construction of current text/image input and validation of known image limits."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .config import Config


class InputError(ValueError):
    """An invalid or unsupported input that can be reported before model execution."""


MAX_IMAGE_BYTES = 32 * 1024 * 1024


def _deepseek_vision(config: Config | None) -> bool:
    return bool(
        config
        and urlsplit(config.model.base_url or "").hostname == "api.deepseek.com"
        and config.model.model in {"deepseek-flash", "deepseek-v4-flash-vision-exp"}
    )


def _decoded_size(encoded: str) -> int:
    padding = 2 if encoded.endswith("==") else int(encoded.endswith("="))
    return len(encoded) * 3 // 4 - padding


def _image_mime(header: bytes) -> str:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    raise InputError("unsupported_image_format: expected JPEG, PNG, GIF, or WebP content")


def load_local_image(path: Path, *, max_encoded_bytes: int | None = None) -> dict:
    """Read a local image with a bounded allocation and return a standard image block."""
    try:
        if not path.is_file():
            raise InputError(f"image_not_found_or_not_regular: {path}")
        with path.open("rb") as handle:
            data = handle.read(MAX_IMAGE_BYTES + 1)
        if not data:
            raise InputError(f"empty_image: {path}")
        if len(data) > MAX_IMAGE_BYTES:
            raise InputError("image_too_large: local image files are limited to 32 MiB")
        if max_encoded_bytes is not None and 4 * ((len(data) + 2) // 3) >= max_encoded_bytes:
            raise InputError("image_payload_too_large: base64 alone reaches the 48 MiB DeepSeek request limit")
        mime = _image_mime(data[:16])
        return {"type": "image", "base64": base64.b64encode(data).decode("ascii"), "mime_type": mime}
    except OSError as exc:
        raise InputError(f"unable_to_read_image: {path}: {exc}") from exc


def build_human_message(
    text: str = "", *, images: Sequence[str] = (), workdir: str | Path | None = None,
    config: Config | None = None,
) -> HumanMessage:
    if not text.strip() and not images:
        raise InputError("empty_input")
    if not images:
        return HumanMessage(content=text)
    blocks = [{"type": "text", "text": text}] if text.strip() else []
    known_limits = _deepseek_vision(config)
    encoded_bytes = 0
    if known_limits and len(images) > 600:
        raise InputError("too_many_images: DeepSeek allows 600 images per request")
    for source in images:
        if not source.strip():
            raise InputError("empty_image_source")
        try:
            parsed = urlsplit(source)
        except ValueError as exc:
            raise InputError(f"invalid_image_url: {exc}") from exc
        if parsed.scheme:
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise InputError("invalid_image_url: use an HTTP(S) URL with a host")
            if known_limits and len(source) > 8192:
                raise InputError("image_url_too_long: DeepSeek allows 8192 characters")
            blocks.append({"type": "image", "url": source})
            continue
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = Path(workdir or Path.cwd()) / path
        block = load_local_image(path, max_encoded_bytes=48 * 1024 * 1024 - encoded_bytes if known_limits else None)
        encoded_bytes += len(block["base64"])
        blocks.append(block)
    message = HumanMessage(content_blocks=blocks)
    validate_image_messages([message], config)
    return message


def validate_human_message(message: HumanMessage) -> None:
    content = message.content
    if isinstance(content, str):
        valid = bool(content.strip())
    else:
        valid = any(
            isinstance(block, dict) and (
                (block.get("type") == "text" and str(block.get("text", "")).strip())
                or (block.get("type") == "image" and (block.get("url") or block.get("base64")))
            )
            for block in content
        )
    if not valid:
        raise ValueError("empty_input")


def summarize_message_content(message: BaseMessage) -> str | list:
    """A trajectory-friendly copy of message content with inline image payloads replaced by digests."""
    content = message.content
    if isinstance(content, str):
        return content
    summary = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image" and block.get("base64"):
            encoded = block["base64"]
            summary.append({
                "type": "image",
                "mime_type": block.get("mime_type"),
                "bytes": _decoded_size(encoded),
                "sha256": hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest(),
            })
        else:
            summary.append(block)
    return summary


def validate_image_messages(messages: Sequence, config: Config | None, profile: dict | None = None) -> None:
    images = [
        block for message in messages if isinstance(message.content, list)
        for block in message.content_blocks if block.get("type") == "image"
    ]
    if not images:
        return
    if config and not config.model.base_url and profile and profile.get("image_inputs") is False:
        raise InputError("image_inputs_not_supported: the selected model does not accept images")
    if not _deepseek_vision(config):
        return
    if len(images) > 600:
        raise InputError("too_many_images: DeepSeek allows 600 images across the conversation")
    total_bytes = 0
    encoded_bytes = 0
    for block in images:
        if len(block.get("url", "")) > 8192:
            raise InputError("image_url_too_long: DeepSeek allows 8192 characters")
        encoded = block.get("base64", "")
        size = _decoded_size(encoded)
        if size > 32 * 1024 * 1024:
            raise InputError("image_too_large: DeepSeek inline images are limited to 32 MiB")
        total_bytes += size
        encoded_bytes += len(encoded)
    if total_bytes > 64 * 1024 * 1024:
        raise InputError("images_too_large: DeepSeek allows 64 MiB of images per request")
    if encoded_bytes >= 48 * 1024 * 1024:
        raise InputError("image_payload_too_large: base64 alone reaches the 48 MiB DeepSeek request limit")


def extract_answer(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    if not messages:
        raise ValueError("agent_returned_no_messages")
    message = messages[-1]
    if not isinstance(message, AIMessage) and not hasattr(message, "content"):
        raise ValueError("agent_returned_invalid_message")
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"} and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""
