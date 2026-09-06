"""The minimal, model-backed Agent Loop."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from .clients import anthropic as anthropic_client
from .clients import openai as openai_client
from .config import Config, ConfigError
from .runtime.session import ErrorInfo, Session, SessionResult


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
    """Small stateful facade around :func:`run` for embedding applications.

    The facade stores only the immutable Config and optional transport-boundary
    test clients; conversation state is created anew for every ``run`` call.
    """

    def __init__(
        self,
        config: Config,
        *,
        anthropic_sdk_client: Any | None = None,
        openai_sdk_client: Any | None = None,
    ):
        self.config = config
        self.anthropic_sdk_client = anthropic_sdk_client
        self.openai_sdk_client = openai_sdk_client

    def run(self, request: InputRequest | Mapping[str, Any]) -> SessionResult | None:
        return run(
            request,
            self.config,
            anthropic_sdk_client=self.anthropic_sdk_client,
            openai_sdk_client=self.openai_sdk_client,
        )


def run(
    request: InputRequest | Mapping[str, Any],
    config: Config,
    *,
    anthropic_sdk_client: Any | None = None,
    openai_sdk_client: Any | None = None,
) -> SessionResult | None:
    """Run one request as one session.

    ``None`` is returned only for the intentionally silent empty-input case.
    Optional client arguments are transport-boundary injection points for tests;
    production callers leave them unset and use the official SDKs.
    """

    try:
        normalized = request if isinstance(request, InputRequest) else InputRequest.model_validate(request)
    except Exception as exc:
        # A malformed non-empty request is a real input/configuration failure;
        # unlike an intentionally empty request it is reported to the caller.
        return _standalone_failure("input_error", str(exc))

    if not normalized.has_input():
        return None

    image_reference = _safe_image_reference(normalized.image)
    try:
        session = Session.start(normalized.query, image_reference, config)
    except Exception as exc:
        return _standalone_failure("runtime_error", f"Unable to create session: {exc}")

    if normalized.query is None:
        return session.finish(
            "not_supported",
            stop_reason="image_only_not_implemented",
        )

    try:
        system_prompt = config.system_prompt()
        if config.model.provider == "anthropic":
            client = (
                anthropic_sdk_client
                if anthropic_sdk_client is not None
                else anthropic_client.create_client(config)
            )
        else:
            client = (
                openai_sdk_client
                if openai_sdk_client is not None
                else openai_client.create_client(config)
            )
    except ConfigError as exc:
        return session.error(ErrorInfo(code=exc.code, message=exc.message))
    except Exception as exc:
        return session.error(
            ErrorInfo(code="model_unavailable", message=_safe_exception_message(exc, config))
        )

    user_message = _build_user_message(
        InputRequest(query=normalized.query, image=image_reference)
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    session.set_messages(messages)

    while session.state.turn < config.loop.max_turns:
        session.state.turn += 1
        session.save_state()
        session.trace.write(
            "model_call_started",
            session_id=session.state.session_id,
            turn=session.state.turn,
            provider=config.model.provider,
            model=config.model.model,
            max_tokens=config.model.max_tokens,
        )
        try:
            if config.model.provider == "anthropic":
                response = anthropic_client.create_message(
                    client, config.model, system_prompt, messages
                )
                assistant_content = list(_field(response, "content", []) or [])
                answer = _extract_anthropic_text(assistant_content)
                tool_calls = [
                    block for block in assistant_content if _block_type(block) == "tool_use"
                ]
                assistant_message = {"role": "assistant", "content": _json_blocks(assistant_content)}
                response_meta = _response_summary(response, provider="anthropic")
            else:
                response = openai_client.create_completion(
                    client, config.model, system_prompt, messages
                )
                choice = _first_openai_choice(response)
                assistant = _field(choice, "message")
                if assistant is None:
                    raise ValueError("response contained no assistant message")
                content = _field(assistant, "content")
                answer = _extract_openai_text(content)
                tool_calls = list(_field(assistant, "tool_calls", []) or [])
                assistant_message = {
                    "role": "assistant",
                    "content": content or "",
                }
                if tool_calls:
                    assistant_message["tool_calls"] = _json_blocks(tool_calls)
                response_meta = _response_summary(response, provider="openai")

            messages.append(assistant_message)
            session.set_messages(messages)
            session.trace.write(
                "model_call_finished",
                session_id=session.state.session_id,
                turn=session.state.turn,
                provider=config.model.provider,
                response=response_meta,
                answer=answer,
                tool_call_count=len(tool_calls),
            )
        except TimeoutError as exc:
            session.trace.write(
                "model_call_finished",
                session_id=session.state.session_id,
                turn=session.state.turn,
                provider=config.model.provider,
                status="failed",
                error={"code": "model_timeout", "message": _safe_exception_message(exc, config)},
            )
            return session.error(
                ErrorInfo(code="model_timeout", message=_safe_exception_message(exc, config))
            )
        except Exception as exc:
            code = _classify_model_exception(exc)
            session.trace.write(
                "model_call_finished",
                session_id=session.state.session_id,
                turn=session.state.turn,
                provider=config.model.provider,
                status="failed",
                error={"code": code, "message": _safe_exception_message(exc, config)},
            )
            return session.error(ErrorInfo(code=code, message=_safe_exception_message(exc, config)))

        if tool_calls:
            return session.error(
                ErrorInfo(
                    code="unsupported_action",
                    message="The model requested a tool, but v0.1.0 has no tools registered.",
                )
            )
        if answer.strip():
            return session.finish("succeeded", answer=answer, stop_reason="answer")
        return session.error(
            ErrorInfo(code="empty_model_response", message="The model returned no text answer.")
        )

    return session.error(
        ErrorInfo(code="budget_exceeded", message="The maximum Agent Loop turns was reached.")
    )


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


def _extract_anthropic_text(blocks: list[Any]) -> str:
    texts: list[str] = []
    for block in blocks:
        if _block_type(block) == "text":
            text = getattr(block, "text", None)
            if text is None and isinstance(block, Mapping):
                text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts).strip()


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


def _block_type(block: Any) -> str | None:
    if isinstance(block, Mapping):
        value = block.get("type")
    else:
        value = getattr(block, "type", None)
    return str(value) if value is not None else None


def _json_blocks(values: Any) -> Any:
    return [_json_value(value) for value in values]


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _response_summary(response: Any, *, provider: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"provider": provider}
    for key in ("id", "model", "stop_reason", "finish_reason", "usage"):
        value = _field(response, key)
        if value is not None:
            summary[key] = _json_value(value)
    if provider == "openai":
        choices = _field(response, "choices", []) or []
        if choices:
            finish = _field(choices[0], "finish_reason")
            if finish is not None:
                summary["finish_reason"] = finish
    return summary


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


def _standalone_failure(code: str, message: str) -> SessionResult:
    return SessionResult(
        session_id="",
        status="failed",
        errors=[ErrorInfo(code=code, message=message)],
        session_path="",
        stop_reason=code,
    )
