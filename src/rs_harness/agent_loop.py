"""The minimal, model-backed Agent Loop."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, field_validator

from .config import Config, ConfigError
from .runtime.images import ImageAttachment
from .runtime.messages import (
    assistant_turn,
    build_user_message_content,
    decode_tool_arguments,
    history_tool_error,
    messages_for_model,
    safe_image_reference,
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


class _LoopFailure(Exception):
    """Internal early-exit from the locked interaction body."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        turn: int,
        session_status: Literal["running", "closed", "failed"] = "running",
        record: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.turn = turn
        self.session_status = session_status
        self.record = record


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

    image_reference = safe_image_reference(normalized.image)
    try:
        session = Session.open(config, session_id)
    except ConfigError as exc:
        return _interaction_failure(session_id, exc.code, exc.message, 0)
    except Exception as exc:
        return _interaction_failure(session_id, "runtime_error", str(exc), 0)

    with session.locked():
        try:
            return _interact(
                session,
                normalized,
                image_reference,
                config,
                openai_sdk_client=openai_sdk_client,
                workdir=workdir,
                tool_log=tool_log,
            )
        except _LoopFailure as failure:
            if failure.record:
                session.add_error(ErrorInfo(code=failure.code, message=failure.message))
            return _interaction_failure(
                session_id,
                failure.code,
                failure.message,
                failure.turn,
                session_status=failure.session_status,
            )


def _interact(
    session: Session,
    normalized: InputRequest,
    image_reference: str | None,
    config: Config,
    *,
    openai_sdk_client: Any | None,
    workdir: str | Path | None,
    tool_log: Callable[[str, str], None] | None,
) -> InteractionResult:
    if session.state.status != "running":
        raise _LoopFailure(
            "session_closed",
            "session is not running",
            turn=session.state.turn,
            session_status=session.state.status,
        )

    try:
        effective_workdir = normalize_workdir(
            workdir if workdir is not None else session.state.workdir
        )
    except Exception as exc:
        raise _LoopFailure("runtime_error", str(exc), turn=session.state.turn, record=True)

    history = session.ledger.messages()
    history_error = history_tool_error(history)
    if history_error is not None:
        raise _LoopFailure(
            "unmatched_tool_call",
            history_error,
            turn=session.state.turn,
            record=True,
        )

    session.state.workdir = effective_workdir
    _ensure_tool_context_unlocked(session, effective_workdir)

    if normalized.query is None:
        try:
            session.append(
                {
                    "role": "user",
                    "content": f"Image reference received: {image_reference}",
                }
            )
        except ValueError as exc:
            raise _LoopFailure(
                str(exc),
                str(exc),
                turn=session.state.turn,
                session_status=session.state.status,
            )
        return InteractionResult(
            session_id=session.state.session_id,
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
        session.append(
            {
                "role": "user",
                "content": build_user_message_content(
                    normalized.query, image_reference, effective_workdir
                ),
            }
        )
    except ConfigError as exc:
        raise _LoopFailure(exc.code, exc.message, turn=session.state.turn)
    except ValueError as exc:
        raise _LoopFailure(
            str(exc),
            str(exc),
            turn=session.state.turn,
            session_status=session.state.status,
        )
    except Exception as exc:
        raise _LoopFailure("runtime_error", str(exc), turn=session.state.turn)

    return _drive_turns(session, config, client, workdir=effective_workdir, tool_log=tool_log)


def _drive_turns(
    session: Session,
    config: Config,
    client: Any,
    *,
    workdir: str,
    tool_log: Callable[[str, str], None] | None,
) -> InteractionResult:
    session_id = session.state.session_id
    messages = session.ledger.messages()
    for _ in range(config.loop.max_turns):
        try:
            turn = session.next_turn()
        except ValueError as exc:
            raise _LoopFailure(
                str(exc),
                str(exc),
                turn=session.state.turn,
                session_status=session.state.status,
            )

        try:
            response = client.chat.completions.create(
                model=config.model.model,
                messages=messages_for_model(messages),
                max_tokens=config.model.max_tokens,
                tools=TOOLS,
            )
            assistant_message, tool_calls, answer = assistant_turn(response)
            session.append(assistant_message)
            messages.append(assistant_message)
        except TimeoutError as exc:
            raise _LoopFailure(
                "model_timeout", _safe_exception_message(exc, config), turn=turn, record=True
            )
        except Exception as exc:
            raise _LoopFailure(
                _classify_model_exception(exc),
                _safe_exception_message(exc, config),
                turn=turn,
                record=True,
            )

        if tool_calls:
            _execute_tool_calls(session, messages, tool_calls, workdir, config, tool_log)
            continue

        if answer.strip():
            return InteractionResult(
                session_id=session_id,
                status="succeeded",
                answer=answer,
                session_status=session.state.status,
                turn=turn,
            )
        raise _LoopFailure(
            "empty_model_response", "The model returned no text answer.", turn=turn, record=True
        )

    raise _LoopFailure(
        "budget_exceeded",
        "The maximum Agent Loop turns was reached.",
        turn=session.state.turn,
        record=True,
    )


def _execute_tool_calls(
    session: Session,
    messages: list[dict[str, Any]],
    tool_calls: list[tuple[str, str, Any]],
    workdir: str,
    config: Config,
    tool_log: Callable[[str, str], None] | None,
) -> None:
    image_attachments: list[ImageAttachment] = []
    for tool_id, tool_name, raw_arguments in tool_calls:
        arguments, argument_error = decode_tool_arguments(raw_arguments)
        if argument_error is not None:
            output = argument_error
        else:
            try:
                raw_output = dispatch_tool(
                    tool_name,
                    arguments,
                    workdir,
                    bash_timeout_s=config.loop.bash_timeout_s,
                )
                if isinstance(raw_output, ImageAttachment):
                    image_attachments.append(raw_output)
                    output = (
                        f"Image loaded: {raw_output.path} "
                        f"({raw_output.mime}, {raw_output.bytes} bytes)."
                    )
                else:
                    output = str(raw_output)
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
        session.append(tool_message)
        messages.append(tool_message)

    if image_attachments:
        paths = "\n".join(f"- {attachment.path}" for attachment in image_attachments)
        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Images loaded by read_image and attached for visual inspection:\n"
                        f"{paths}"
                    ),
                },
                *(attachment.ref_block() for attachment in image_attachments),
            ],
        }
        session.append(image_message)
        messages.append(image_message)


_TOOL_CONTEXT_MARKER = "RS-Harness tool context v1"


def _ensure_tool_context_unlocked(session: Session, workdir: str) -> None:
    """Add a deterministic capability/workspace note once per workspace."""

    content = (
        f"{_TOOL_CONTEXT_MARKER}\n"
        "Available tools: bash, read_file, write_file, edit_file, glob, read_image.\n"
        "Use read_image for local image inspection; use glob first when the image path is unclear.\n"
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
    session.append({"role": "system", "content": content})


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
        image=safe_image_reference(image),
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
