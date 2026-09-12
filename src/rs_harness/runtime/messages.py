"""Stateless shaping of OpenAI-protocol messages for the agent loop."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .images import MAX_IMAGE_BYTES, inspect_local_image, sniff_mime
from .trace import _json_safe

__all__ = [
    "assistant_turn",
    "build_user_message_content",
    "classify_image_reference",
    "decode_tool_arguments",
    "extract_openai_text",
    "field",
    "first_openai_choice",
    "history_tool_error",
    "image_ref_block_to_request",
    "messages_for_model",
    "safe_image_reference",
    "tool_call_id",
    "validated_tool_calls",
]


def history_tool_error(messages: list[dict[str, Any]]) -> str | None:
    """Detect tool calls from an earlier run that cannot be safely replayed."""

    pending: dict[str, bool] = {}
    for message in messages:
        if not isinstance(message, Mapping):
            return "Session contains a malformed message; start a new session."
        role = message.get("role")
        if role == "assistant":
            if pending:
                ids = ", ".join(pending)
                return f"Session has a message before tool result(s) for {ids}; start a new session."
            raw_calls_value = message.get("tool_calls")
            if raw_calls_value is None:
                raw_calls = []
            elif isinstance(raw_calls_value, (list, tuple)):
                raw_calls = raw_calls_value
            else:
                return "Session contains malformed tool calls; start a new session."
            for call in raw_calls:
                call_id = tool_call_id(call)
                if call_id is None:
                    return "Session contains a tool call without an id; start a new session."
                call_type = field(call, "type")
                if call_type not in (None, "function"):
                    return f"Session contains unsupported tool call type for {call_id}; start a new session."
                function = field(call, "function")
                name = field(function, "name", field(call, "name"))
                if not isinstance(name, str) or not name.strip():
                    return f"Session contains malformed tool call {call_id}; start a new session."
                if call_id in pending:
                    return f"Session contains duplicate tool call id {call_id}; start a new session."
                pending[call_id] = True
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                return "Session contains a tool result without an id; start a new session."
            if not isinstance(message.get("content"), str):
                return "Session contains a tool result with invalid content; start a new session."
            if call_id not in pending:
                return f"Session contains an unexpected tool result for {call_id}; start a new session."
            del pending[call_id]
        elif pending:
            ids = ", ".join(pending)
            return f"Session has a message before tool result(s) for {ids}; start a new session."
    if pending:
        ids = ", ".join(pending)
        return f"Session contains unmatched tool call(s): {ids}. Start a new session; historical tools were not replayed."
    return None


def tool_call_id(call: Any) -> str | None:
    value = field(call, "id")
    return value if isinstance(value, str) and value.strip() else None


def validated_tool_calls(raw_calls: list[Any]) -> list[tuple[str, str, Any]]:
    """Validate response call identities before persisting or executing them."""

    if not isinstance(raw_calls, list):
        raise ValueError("response tool_calls was not a list")
    # OpenAI requires IDs to be unique within one assistant response. IDs from
    # already completed turns may be reused by a provider, so they are not
    # treated as a new response error.
    seen: set[str] = set()
    validated: list[tuple[str, str, Any]] = []
    for call in raw_calls:
        call_id = tool_call_id(call)
        if call_id is None:
            raise ValueError("response contained a tool call without an id")
        if call_id in seen:
            raise ValueError(f"response contained duplicate tool call id: {call_id}")
        call_type = field(call, "type")
        if call_type not in (None, "function"):
            raise ValueError(f"tool call {call_id} had unsupported type")
        function = field(call, "function")
        name = field(function, "name", field(call, "name"))
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tool call {call_id} contained no function name")
        arguments = field(function, "arguments", field(call, "arguments"))
        validated.append((call_id, name.strip(), arguments))
        seen.add(call_id)
    return validated


def decode_tool_arguments(raw_arguments: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw_arguments, str):
        try:
            decoded = json.loads(raw_arguments)
        except (TypeError, ValueError) as exc:
            return {}, f"Error: Invalid JSON arguments: {str(exc)[:500]}"
    elif isinstance(raw_arguments, Mapping):
        decoded = dict(raw_arguments)
    else:
        return {}, "Error: Invalid JSON arguments: expected an object"
    if not isinstance(decoded, dict):
        return {}, "Error: Invalid JSON arguments: expected an object"
    return decoded, None


def build_user_message_content(
    query: str,
    image_reference: str | None,
) -> Any:
    """Build the user content, attaching directly viewable images as blocks.

    Returns a plain string when no image attaches (backward compatible), or a
    text block plus one block per image. Image bytes never enter the ledger:
    local files are persisted as ``image_ref`` evidence and rebuilt as
    ``image_url`` data URLs at request time.
    """

    if image_reference is None:
        return query

    block, reason = classify_image_reference(image_reference)
    if block is None:
        return (
            f"{query}\nImage reference received: {image_reference} "
            f"(not directly viewable: {reason}). "
            "RS-Harness cannot view this image; do not guess its content. "
            "Remote-sensing analysis requires MCP tools."
        )
    text = f"{query}\nImage attached: {image_reference}"
    return [{"type": "text", "text": text}, block]


def classify_image_reference(reference: str) -> tuple[dict[str, Any] | None, str]:
    """Turn one image reference into a ledger block or an opaque reason."""

    lowered = reference.lower()
    if lowered.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": reference}}, ""
    if reference.startswith("[image reference omitted"):
        return None, "inline image bytes were not accepted; pass a file path or URL"
    if "://" in lowered:
        return None, "remote object reference; fetching requires MCP tools"
    attachment, reason = inspect_local_image(reference)
    if attachment is None:
        return None, reason
    return attachment.ref_block(), ""


def safe_image_reference(image: str | None) -> str | None:
    """Keep image inputs as references; never put inline image bytes in a prompt.

    Ordinary paths/URIs remain useful to the model as opaque identifiers. Data
    URIs and unusually long values are replaced by a short deterministic
    summary so a caller cannot accidentally upload image bytes or flood a
    session trace.
    """

    if image is None:
        return None
    if image.lower().startswith("data:") or len(image) > 512:
        digest = hashlib.sha256(image.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"[image reference omitted; length={len(image)} sha256={digest}]"
    return image


def extract_openai_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, Mapping):
                text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
        return "\n".join(texts).strip()
    return ""


def first_openai_choice(response: Any) -> Any:
    choices = field(response, "choices")
    if not choices:
        raise ValueError("response contained no choices")
    return choices[0]


def field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def messages_for_model(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep provider output metadata in the ledger but send request fields only.

    Ledger ``image_ref`` blocks are rebuilt as ``image_url`` data URLs here so
    image bytes are re-attached from disk on every request and never persisted.
    """

    assistant_fields = {
        "role",
        "content",
        "name",
        "tool_calls",
        "refusal",
        "audio",
    }
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant":
            result.append({key: value for key, value in message.items() if key in assistant_fields})
        else:
            converted = dict(message)
            content = converted.get("content")
            if isinstance(content, list):
                converted["content"] = [
                    image_ref_block_to_request(block)
                    if isinstance(block, Mapping) and block.get("type") == "image_ref"
                    else block
                    for block in content
                ]
            result.append(converted)
    return result


def image_ref_block_to_request(block: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild one ledger ``image_ref`` block as a request-time block."""

    path = block.get("path")
    raw: bytes | None = None
    if isinstance(path, str):
        try:
            raw = Path(path).read_bytes()
        except OSError:
            raw = None
    mime = sniff_mime(raw[:16]) if raw is not None else None
    if raw is None or mime is None or len(raw) > MAX_IMAGE_BYTES:
        return {
            "type": "text",
            "text": f"Image previously attached at {path} is no longer available.",
        }
    encoded = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def assistant_turn(response: Any) -> tuple[dict[str, Any], list[tuple[str, str, Any]], str]:
    """Extract the persisted assistant message, its validated calls, and text."""

    choice = first_openai_choice(response)
    assistant = field(choice, "message")
    if assistant is None:
        raise ValueError("response contained no assistant message")
    raw_tool_calls_value = field(assistant, "tool_calls")
    if raw_tool_calls_value is None:
        raw_tool_calls = []
    elif isinstance(raw_tool_calls_value, (list, tuple)):
        raw_tool_calls = list(raw_tool_calls_value)
    else:
        raise ValueError("response tool_calls was not a list")
    tool_calls = validated_tool_calls(raw_tool_calls)
    assistant_message = _json_safe(assistant)
    if not isinstance(assistant_message, dict):
        raise ValueError("response message was not an object")
    if assistant_message.get("role") is None:
        assistant_message["role"] = "assistant"
    elif assistant_message.get("role") != "assistant":
        raise ValueError("response message had an invalid role")
    answer = extract_openai_text(field(assistant, "content"))
    return assistant_message, tool_calls, answer
