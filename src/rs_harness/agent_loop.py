"""The minimal, model-backed Agent Loop."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, field_validator

from .config import Config, ConfigError
from .runtime.images import (
    MAX_IMAGE_BYTES,
    extract_image_paths,
    inspect_local_image,
    sniff_mime,
)
from .runtime.session import ErrorInfo, InteractionResult, Session, SessionResult
from .runtime.tools import TOOLS, dispatch_tool, normalize_workdir

__all__ = [
    "AgentLoop",
    "InputRequest",
    "run",
    "create_session",
    "close_session",
]


class InputRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = None
    image: str | None = None

    @field_validator("query", "image", mode="before")
    @classmethod
    def blank_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("query and image must be strings")
        value = value.strip()
        return value or None

    def has_input(self) -> bool:
        return self.query is not None or self.image is not None


class AgentLoop:
    """Facade for running interactions against persistent sessions."""

    def __init__(
        self,
        config: Config,
        *,
        openai_sdk_client: Any | None = None,
    ):
        self.config = config
        self.openai_sdk_client = openai_sdk_client

    def run(
        self,
        session_id: str,
        request: InputRequest | Mapping[str, Any],
        *,
        workdir: str | Path | None = None,
        tool_log: Callable[[str, str], None] | None = None,
    ) -> InteractionResult:
        return run(
            session_id,
            request,
            self.config,
            openai_sdk_client=self.openai_sdk_client,
            workdir=workdir,
            tool_log=tool_log,
        )


def run(
    session_id: str,
    request: InputRequest | Mapping[str, Any],
    config: Config,
    *,
    openai_sdk_client: Any | None = None,
    workdir: str | Path | None = None,
    tool_log: Callable[[str, str], None] | None = None,
) -> InteractionResult:
    """Append one interaction to an existing session and run the model.

    ``tool_log(name, preview)`` receives each tool call name and the first 200
    characters of its (already redacted) result; the loop itself stays silent.
    """

    try:
        normalized = request if isinstance(request, InputRequest) else InputRequest.model_validate(request)
    except Exception as exc:
        # A malformed non-empty request is a real input/configuration failure;
        # unlike an intentionally empty request it is reported to the caller.
        return _interaction_failure(session_id, "input_error", str(exc), 0)

    if not normalized.has_input():
        return _interaction_failure(session_id, "empty_input", "request has no input", 0)

    image_reference = _safe_image_reference(normalized.image)
    try:
        session = Session.open(config, session_id)
    except ConfigError as exc:
        return _interaction_failure(session_id, exc.code, exc.message, 0)
    except Exception as exc:
        return _interaction_failure(session_id, "runtime_error", str(exc), 0)

    with session.locked():
        if session.state.status != "running":
            return _interaction_failure(
                session_id,
                "session_closed",
                "session is not running",
                session.state.turn,
                session_status=session.state.status,
            )

        try:
            effective_workdir = normalize_workdir(
                workdir if workdir is not None else session.state.workdir
            )
        except Exception as exc:
            session.add_error(ErrorInfo(code="runtime_error", message=str(exc)))
            return _interaction_failure(session_id, "runtime_error", str(exc), session.state.turn)

        history = session.ledger.messages()
        history_error = _history_tool_error(history)
        if history_error is not None:
            session.add_error(ErrorInfo(code="unmatched_tool_call", message=history_error))
            return _interaction_failure(
                session_id,
                "unmatched_tool_call",
                history_error,
                session.state.turn,
            )

        session.state.workdir = effective_workdir
        _ensure_tool_context_unlocked(session, effective_workdir)

        if normalized.query is None:
            try:
                session._append_message_unlocked(
                    {
                        "role": "user",
                        "content": f"Image reference received: {image_reference}",
                    }
                )
            except ValueError as exc:
                return _interaction_failure(
                    session_id,
                    str(exc),
                    str(exc),
                    session.state.turn,
                    session_status=session.state.status,
                )
            return InteractionResult(
                session_id=session_id,
                status="not_supported",
                session_status=session.state.status,
                turn=session.state.turn,
            )

        try:
            if session.state.query is None:
                session.state.query = normalized.query
            if openai_sdk_client is not None:
                client = openai_sdk_client
            else:
                kwargs: dict[str, Any] = {
                    "api_key": config.api_key(),
                    "timeout": config.model.timeout_s,
                }
                if config.model.base_url:
                    kwargs["base_url"] = config.model.base_url
                client = OpenAI(**kwargs)
            session._append_message_unlocked(
                {
                    "role": "user",
                    "content": _build_user_message_content(
                        normalized.query, image_reference, effective_workdir
                    ),
                }
            )
        except ConfigError as exc:
            return _interaction_failure(session_id, exc.code, exc.message, session.state.turn)
        except ValueError as exc:
            return _interaction_failure(
                session_id,
                str(exc),
                str(exc),
                session.state.turn,
                session_status=session.state.status,
            )
        except Exception as exc:
            return _interaction_failure(session_id, "runtime_error", str(exc), session.state.turn)

        messages = session.ledger.messages()
        for _ in range(config.loop.max_turns):
            try:
                turn = session.next_turn()
            except ValueError as exc:
                return _interaction_failure(
                    session_id,
                    str(exc),
                    str(exc),
                    session.state.turn,
                    session_status=session.state.status,
                )

            try:
                response = client.chat.completions.create(
                    model=config.model.model,
                    messages=_messages_for_model(messages),
                    max_tokens=config.model.max_tokens,
                    tools=TOOLS,
                )
                choice = _first_openai_choice(response)
                assistant = _field(choice, "message")
                if assistant is None:
                    raise ValueError("response contained no assistant message")
                raw_tool_calls_value = _field(assistant, "tool_calls")
                if raw_tool_calls_value is None:
                    raw_tool_calls = []
                elif isinstance(raw_tool_calls_value, (list, tuple)):
                    raw_tool_calls = list(raw_tool_calls_value)
                else:
                    raise ValueError("response tool_calls was not a list")
                tool_calls = _validated_tool_calls(raw_tool_calls)
                assistant_message = _json_value(assistant)
                if not isinstance(assistant_message, dict):
                    raise ValueError("response message was not an object")
                if assistant_message.get("role") is None:
                    assistant_message["role"] = "assistant"
                elif assistant_message.get("role") != "assistant":
                    raise ValueError("response message had an invalid role")
                session._append_message_unlocked(assistant_message)
                messages.append(assistant_message)
                content = _field(assistant, "content")
                answer = _extract_openai_text(content)
            except TimeoutError as exc:
                error = ErrorInfo(code="model_timeout", message=_safe_exception_message(exc, config))
                session.add_error(error)
                return InteractionResult(session_id=session_id, status="failed", errors=[error], session_status=session.state.status, turn=turn)
            except Exception as exc:
                code = _classify_model_exception(exc)
                error = ErrorInfo(code=code, message=_safe_exception_message(exc, config))
                session.add_error(error)
                return InteractionResult(session_id=session_id, status="failed", errors=[error], session_status=session.state.status, turn=turn)

            if tool_calls:
                for tool_id, tool_name, raw_arguments in tool_calls:
                    arguments, argument_error = _decode_tool_arguments(raw_arguments)
                    if argument_error is not None:
                        output = argument_error
                    else:
                        try:
                            output = dispatch_tool(
                                tool_name,
                                arguments,
                                effective_workdir,
                                bash_timeout_s=config.loop.bash_timeout_s,
                            )
                        except Exception as exc:  # defensive: a tool must never break the loop
                            output = f"Error: Tool execution failed: {exc}"
                    # A tool observation (e.g. ``env`` output) can echo the API
                    # key; redact before the value reaches the log, the ledger,
                    # or the next model request.
                    output = _redact_secrets(str(output), config)
                    if tool_log is not None:
                        tool_log(tool_name, output[:200])
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": output,
                    }
                    session._append_message_unlocked(tool_message)
                    messages.append(tool_message)
                continue

            if answer.strip():
                return InteractionResult(
                    session_id=session_id,
                    status="succeeded",
                    answer=answer,
                    session_status=session.state.status,
                    turn=turn,
                )
            error = ErrorInfo(code="empty_model_response", message="The model returned no text answer.")
            session.add_error(error)
            return InteractionResult(
                session_id=session_id,
                status="failed",
                errors=[error],
                session_status=session.state.status,
                turn=turn,
            )

        error = ErrorInfo(code="budget_exceeded", message="The maximum Agent Loop turns was reached.")
        session.add_error(error)
        return InteractionResult(
            session_id=session_id,
            status="failed",
            errors=[error],
            session_status=session.state.status,
            turn=session.state.turn,
        )


_TOOL_CONTEXT_MARKER = "RS-Harness tool context v1"


def _ensure_tool_context_unlocked(session: Session, workdir: str) -> None:
    """Add a deterministic capability/workspace note once per workspace."""

    content = (
        f"{_TOOL_CONTEXT_MARKER}\n"
        "Available tools: bash, read_file, write_file, edit_file, glob.\n"
        f"Working directory: {workdir}\n"
        "File-tool paths stay inside this directory. Tool results are observations; "
        "remote-sensing semantics remain behind MCP."
    )
    for message in reversed(session.ledger.messages()):
        if message.get("role") != "system":
            continue
        if isinstance(message.get("content"), str) and message["content"].startswith(
            f"{_TOOL_CONTEXT_MARKER}\n"
        ):
            if message.get("content") == content:
                return
            break
    session._append_message_unlocked({"role": "system", "content": content})


def _history_tool_error(messages: list[dict[str, Any]]) -> str | None:
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
                call_id = _tool_call_id(call)
                if call_id is None:
                    return "Session contains a tool call without an id; start a new session."
                call_type = _field(call, "type")
                if call_type not in (None, "function"):
                    return f"Session contains unsupported tool call type for {call_id}; start a new session."
                function = _field(call, "function")
                name = _field(function, "name", _field(call, "name"))
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


def _tool_call_id(call: Any) -> str | None:
    value = _field(call, "id")
    return value if isinstance(value, str) and value.strip() else None


def _validated_tool_calls(raw_calls: list[Any]) -> list[tuple[str, str, Any]]:
    """Validate response call identities before persisting or executing them."""

    if not isinstance(raw_calls, list):
        raise ValueError("response tool_calls was not a list")
    # OpenAI requires IDs to be unique within one assistant response. IDs from
    # already completed turns may be reused by a provider, so they are not
    # treated as a new response error.
    seen: set[str] = set()
    validated: list[tuple[str, str, Any]] = []
    for call in raw_calls:
        call_id = _tool_call_id(call)
        if call_id is None:
            raise ValueError("response contained a tool call without an id")
        if call_id in seen:
            raise ValueError(f"response contained duplicate tool call id: {call_id}")
        call_type = _field(call, "type")
        if call_type not in (None, "function"):
            raise ValueError(f"tool call {call_id} had unsupported type")
        function = _field(call, "function")
        name = _field(function, "name", _field(call, "name"))
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tool call {call_id} contained no function name")
        arguments = _field(function, "arguments", _field(call, "arguments"))
        validated.append((call_id, name.strip(), arguments))
        seen.add(call_id)
    return validated


def _decode_tool_arguments(raw_arguments: Any) -> tuple[dict[str, Any], str | None]:
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


def _build_user_message_content(
    query: str,
    image_reference: str | None,
    workdir: str | None,
) -> Any:
    """Build the user content, attaching directly viewable images as blocks.

    Returns a plain string when no image attaches (backward compatible), or a
    text block plus one block per image. Image bytes never enter the ledger:
    local files are persisted as ``image_ref`` evidence and rebuilt as
    ``image_url`` data URLs at request time.
    """

    candidates: list[str] = []
    if image_reference is not None:
        candidates.append(image_reference)
    candidates.extend(extract_image_paths(query, base_dir=workdir))
    seen: set[str] = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]
    if not ordered:
        return query

    blocks: list[dict[str, Any]] = []
    attached: list[str] = []
    opaque: list[tuple[str, str]] = []
    for candidate in ordered:
        block, reason = _classify_image_reference(candidate)
        if block is not None:
            blocks.append(block)
            attached.append(candidate)
        else:
            opaque.append((candidate, reason))
    text = query
    for path in attached:
        text += f"\nImage attached: {path}"
    for path, reason in opaque:
        text += (
            f"\nImage reference received: {path} (not directly viewable: {reason}). "
            "RS-Harness cannot view this image; do not guess its content. "
            "Remote-sensing analysis requires MCP tools."
        )
    if not blocks:
        return text
    return [{"type": "text", "text": text}, *blocks]


def _classify_image_reference(reference: str) -> tuple[dict[str, Any] | None, str]:
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


def _safe_image_reference(image: str | None) -> str | None:
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


def _extract_openai_text(content: Any) -> str:
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


def _first_openai_choice(response: Any) -> Any:
    choices = _field(response, "choices")
    if not choices:
        raise ValueError("response contained no choices")
    return choices[0]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _messages_for_model(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep provider output metadata in the ledger but send request fields only.

    Ledger ``image_ref`` blocks are rebuilt as ``image_url`` data URLs here so
    image bytes are re-attached from disk on every request and never persisted.
    """

    assistant_fields = {
        "role",
        "content",
        "name",
        "tool_calls",
        "function_call",
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
                    _image_ref_block_to_request(block)
                    if isinstance(block, Mapping) and block.get("type") == "image_ref"
                    else block
                    for block in content
                ]
            result.append(converted)
    return result


def _image_ref_block_to_request(block: Mapping[str, Any]) -> dict[str, Any]:
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


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_value(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_value(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {str(key): _json_value(item) for key, item in vars(value).items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _classify_model_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    if "timeout" in name or "timedout" in name:
        return "model_timeout"
    if "response" in name or "parse" in name or isinstance(
        exc, (AttributeError, IndexError, KeyError, TypeError, ValueError)
    ):
        return "model_response_error"
    return "model_unavailable"


def _redact_secrets(text: str, config: Config | None) -> str:
    """Replace the configured API key value wherever it appears in text."""

    if config is None:
        return text
    secret = os.getenv(config.model.api_key_env, "")
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def _safe_exception_message(exc: Exception, config: Config | None = None) -> str:
    # SDK exceptions can echo request headers. Keep the user-facing message
    # short and never include values of configured secret variables.
    message = str(exc).strip() or type(exc).__name__
    return _redact_secrets(message, config)[:1000]


def create_session(
    config: Config,
    *,
    query: str | None = None,
    image: str | None = None,
    workdir: str | Path | None = None,
) -> Session:
    resolved_workdir = normalize_workdir(workdir)
    return Session.create(
        config,
        query=query,
        image=_safe_image_reference(image),
        workdir=resolved_workdir,
    )


def close_session(config: Config, session_id: str, *, status: str = "closed") -> SessionResult:
    return Session.open(config, session_id).close(status)  # type: ignore[arg-type]


def _interaction_failure(
    session_id: str,
    code: str,
    message: str,
    turn: int,
    *,
    session_status: Literal["running", "closed", "failed"] = "running",
) -> InteractionResult:
    error = ErrorInfo(code=code, message=message)
    return InteractionResult(
        session_id=session_id,
        status="failed",
        errors=[error],
        session_status=session_status,
        turn=turn,
    )
