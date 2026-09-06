"""The minimal, model-backed Agent Loop."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, field_validator

from .config import Config, ConfigError
from .runtime.session import ErrorInfo, InteractionResult, Session, SessionResult


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

    def run(self, session_id: str, request: InputRequest | Mapping[str, Any]) -> InteractionResult:
        return run(
            session_id,
            request,
            self.config,
            openai_sdk_client=self.openai_sdk_client,
        )


def run(
    session_id: str,
    request: InputRequest | Mapping[str, Any],
    config: Config,
    *,
    openai_sdk_client: Any | None = None,
) -> InteractionResult:
    """Append one interaction to an existing session and run the model."""

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
        if normalized.query is None:
            with session.locked():
                session._append_message_unlocked({"role": "user", "content": f"Image reference received: {image_reference}"})
            return InteractionResult(session_id=session_id, status="not_supported", session_status="running", turn=session.state.turn)
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
    except ConfigError as exc:
        return _interaction_failure(session_id, exc.code, exc.message, 0)
    except Exception as exc:
        return _interaction_failure(session_id, "runtime_error", str(exc), 0)

    user_message = _build_user_message(
        InputRequest(query=normalized.query, image=image_reference)
    )
    with session.locked():
        try:
            if session.state.query is None and normalized.query is not None:
                session.state.query = normalized.query
            session._append_message_unlocked({"role": "user", "content": user_message})
        except ValueError as exc:
            return _interaction_failure(
                session_id,
                str(exc),
                str(exc),
                session.state.turn,
                session_status=session.state.status,
            )
        messages = session.ledger.messages()
        for _ in range(config.loop.max_turns):
            turn = session.next_turn()
            try:
                response = client.chat.completions.create(
                    model=config.model.model,
                    messages=list(messages),
                    max_tokens=config.model.max_tokens,
                    tools=[],
                )
                choice = _first_openai_choice(response)
                assistant = _field(choice, "message")
                if assistant is None:
                    raise ValueError("response contained no assistant message")
                assistant_message = _json_value(assistant)
                if not isinstance(assistant_message, dict):
                    raise ValueError("response message was not an object")
                assistant_message.setdefault("role", "assistant")
                session._append_message_unlocked(assistant_message)
                messages.append(assistant_message)
                content = _field(assistant, "content")
                answer = _extract_openai_text(content)
                tool_calls = list(_field(assistant, "tool_calls", []) or [])
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
                error = ErrorInfo(code="unsupported_action", message="The model requested a tool, but no tools are registered.")
                session.add_error(error)
                return InteractionResult(session_id=session_id, status="failed", errors=[error], session_status=session.state.status, turn=turn)
            if answer.strip():
                return InteractionResult(session_id=session_id, status="succeeded", answer=answer, session_status=session.state.status, turn=turn)
            error = ErrorInfo(code="empty_model_response", message="The model returned no text answer.")
            session.add_error(error)
            return InteractionResult(session_id=session_id, status="failed", errors=[error], session_status=session.state.status, turn=turn)

        error = ErrorInfo(code="budget_exceeded", message="The maximum Agent Loop turns was reached.")
        session.add_error(error)
        return InteractionResult(session_id=session_id, status="failed", errors=[error], session_status=session.state.status, turn=session.state.turn)


def _build_user_message(request: InputRequest) -> str:
    assert request.query is not None
    if request.image is None:
        return request.query
    return (
        f"{request.query}\n\n"
        "Image reference received: "
        f"{request.image}\n"
        "Limitation: RS-Harness v0.1.0 has not established image facts through MCP. "
        "Do not infer scene content, measurements, targets, or algorithm results from this reference."
    )


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


def _safe_exception_message(exc: Exception, config: Config | None = None) -> str:
    # SDK exceptions can echo request headers. Keep the user-facing message
    # short and never include values of configured secret variables.
    message = str(exc).strip() or type(exc).__name__
    if config is not None:
        import os

        secret = os.getenv(config.model.api_key_env, "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:1000]


def create_session(config: Config, *, query: str | None = None, image: str | None = None) -> Session:
    return Session.create(config, query=query, image=_safe_image_reference(image))


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
