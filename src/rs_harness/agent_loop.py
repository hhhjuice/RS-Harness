"""The minimal, model-backed Agent Loop."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, field_validator

from .config import Config, ConfigError
from .runtime.images import ImageAttachment
from .runtime.background import CONSOLE_LOCK, get_background_task_manager
from .runtime.hooks import HookContext, HookDecision, HookEvent, HookRegistry, PermissionCallback, PermissionRequest, default_hooks
from .runtime.messages import (
    assistant_turn,
    build_user_message_content,
    decode_tool_arguments,
    history_tool_error,
    messages_for_model,
    safe_image_reference,
)
from .runtime.redact import redact_text, secret_redactor
from .runtime.session import (
    ErrorInfo,
    InteractionResult,
    Session,
    SessionResult,
)
from .runtime.tools import SUB_TOOLS, TOOLS, dispatch_tool, normalize_workdir
from .runtime.todos import TodoItem, TodoManager, is_error_observation

__all__ = [
    "InputRequest",
    "run",
    "create_session",
    "close_session",
    "HookContext",
    "HookDecision",
    "HookEvent",
    "HookRegistry",
    "PermissionCallback",
    "PermissionRequest",
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
    console: Callable[[str], None] | None = None,
    hooks: HookRegistry | None = None,
    permission_callback: PermissionCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> InteractionResult:
    """Append one interaction to an existing session and run the model.

    ``console(message)`` receives one ``[HOOK]``-prefixed line per user-facing
    hook notice (working directory, tool invocations, session summary); the
    loop itself stays silent.
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
        registry = default_hooks(
            permission_callback=permission_callback,
            console=console,
            event_sink=session.append_event,
            redactor=secret_redactor(os.getenv(config.model.api_key_env, "")),
            custom=hooks,
        )
        result: InteractionResult | None = None
        try:
            if session.state.status != "running":
                raise _LoopFailure(
                    "session_closed",
                    "session is not running",
                    turn=session.state.turn,
                    session_status=session.state.status,
                )

            try:
                effective_workdir = normalize_workdir(session.state.workdir)
                requested_workdir = None if workdir is None else normalize_workdir(workdir)
            except Exception as exc:
                raise _LoopFailure("runtime_error", str(exc), turn=session.state.turn, record=True)
            if requested_workdir is not None and requested_workdir != effective_workdir:
                raise _LoopFailure(
                    "workdir_mismatch",
                    "session is bound to a different workdir",
                    turn=session.state.turn,
                )

            history = session.ledger.messages()
            history_error = history_tool_error(history)
            if history_error is not None:
                raise _LoopFailure(
                    "unmatched_tool_call",
                    history_error,
                    turn=session.state.turn,
                    record=True,
                )

            registry.trigger(HookContext(
                event="UserPromptSubmit", session=session, session_id=session.state.session_id,
                turn=session.state.turn, query=normalized.query, workdir=effective_workdir,
            ))

            if normalized.query is None:
                _append_user_message(session, f"Image reference received: {image_reference}")
                result = InteractionResult(
                    session_id=session.state.session_id,
                    status="not_supported",
                    session_status=session.state.status,
                    turn=session.state.turn,
                )
                return result

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

            _append_user_message(
                session, build_user_message_content(normalized.query, image_reference)
            )
            # A subagent session (parent_session_id set) runs the same loop with
            # every tool except task, so delegation stays one level deep.
            is_root = session.state.parent_session_id is None
            task_runner = (
                _task_runner(
                    config, client,
                    workdir=effective_workdir,
                    parent_session_id=session.state.session_id,
                    permission_callback=permission_callback,
                    custom_hooks=hooks,
                    console=console,
                )
                if is_root
                else None
            )
            result = _drive_turns(
                session, config, client, workdir=effective_workdir, hooks=registry,
                query=normalized.query,
                tools=TOOLS if is_root else SUB_TOOLS,
                task_runner=task_runner,
                cancel_check=cancel_check,
            )
            return result
        except _LoopFailure as failure:
            if failure.record:
                session.add_error(ErrorInfo(code=failure.code, message=failure.message))
            result = _interaction_failure(
                session_id,
                failure.code,
                failure.message,
                failure.turn,
                session_status=failure.session_status,
            )
            return result
        finally:
            # Success exits already ran Stop inside the turn loop, where its
            # force-continue result can extend the loop; every other exit is
            # reported here exactly once.
            if result is None or result.status != "succeeded":
                registry.trigger(HookContext(
                    event="Stop", session=session, session_id=session.state.session_id,
                    turn=session.state.turn, query=normalized.query,
                    workdir=session.state.workdir or "",
                    metadata={"status": result.status if result is not None else "failed"},
                ))


def _append_user_message(session: Session, content: Any) -> None:
    try:
        session.append({"role": "user", "content": content})
    except ValueError as exc:
        raise _LoopFailure(
            str(exc),
            str(exc),
            turn=session.state.turn,
            session_status=session.state.status,
        )


def _drive_turns(
    session: Session,
    config: Config,
    client: Any,
    *,
    workdir: str,
    hooks: HookRegistry,
    query: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    task_runner: Callable[..., str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> InteractionResult:
    session_id = session.state.session_id
    messages = session.ledger.messages()
    effective_tools = TOOLS if tools is None else tools
    todo_manager = TodoManager([item.model_dump(mode="python") for item in session.state.todos])
    rounds_since_todo = 0
    for _ in range(config.loop.max_turns):
        if cancel_check is not None and cancel_check():
            raise _LoopFailure(
                "cancelled", "Subagent cancellation requested.",
                turn=session.state.turn, record=True,
            )
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
                tools=effective_tools,
            )
            assistant_message, tool_calls, answer = assistant_turn(
                response, redactor=secret_redactor(os.getenv(config.model.api_key_env, ""))
            )
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
            used_todo = _execute_tool_calls(
                session, messages, tool_calls, workdir, config, hooks,
                todo_manager=todo_manager, task_runner=task_runner,
            )
            if cancel_check is not None and cancel_check():
                raise _LoopFailure(
                    "cancelled", "Subagent cancellation requested.",
                    turn=session.state.turn, record=True,
                )
            rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
            if rounds_since_todo >= 3:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "<reminder>Update your todos.</reminder>"}],
                })
                rounds_since_todo = 0
            continue

        if answer.strip():
            force = hooks.trigger(HookContext(
                event="Stop", session=session, session_id=session_id,
                turn=turn, query=query, workdir=workdir,
                metadata={"status": "succeeded"},
            ))
            if force:
                # A Stop hook demanded more work; its message continues the
                # same interaction as a user turn.
                continuation = {"role": "user", "content": force}
                session.append(continuation)
                messages.append(continuation)
                continue
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
    hooks: HookRegistry,
    *,
    todo_manager: TodoManager | None = None,
    task_runner: Callable[..., str] | None = None,
) -> bool:
    image_attachments: list[ImageAttachment] = []
    used_todo = False
    for tool_id, tool_name, raw_arguments in tool_calls:
        arguments, argument_error = decode_tool_arguments(raw_arguments)
        if argument_error is not None:
            output = argument_error
            decision = None
        else:
            context = HookContext(
                event="PreToolUse", session=session, session_id=session.state.session_id,
                turn=session.state.turn, workdir=workdir, tool_call_id=tool_id,
                tool_name=tool_name, arguments=arguments,
            )
            decision = hooks.trigger(context)
            if decision is not None and not decision.allowed:
                output = decision.reason or "Permission denied"
                post_context = HookContext(
                    event="PostToolUse", session=session, session_id=session.state.session_id,
                    turn=session.state.turn, workdir=workdir, tool_call_id=tool_id,
                    tool_name=tool_name, arguments=arguments, output=output,
                    metadata={"skipped": True, "permission_denied": True},
                )
                hooks.trigger(post_context)
                _append_tool_observation(session, messages, tool_id, post_context.output or output)
                continue
            try:
                raw_output = dispatch_tool(
                    tool_name,
                    arguments,
                    workdir,
                    bash_timeout_s=config.loop.bash_timeout_s,
                    allow_workspace_escape=bool(decision and decision.allow_workspace_escape),
                    todo_manager=todo_manager,
                    task_runner=task_runner,
                )
                if tool_name == "todo_write" and not is_error_observation(str(raw_output)):
                    used_todo = True
                if isinstance(raw_output, ImageAttachment):
                    image_attachments.append(raw_output)
                    output = (
                        f"Image loaded: {raw_output.path} "
                        f"({raw_output.mime}, {raw_output.bytes} bytes)."
                    )
                else:
                    output = str(raw_output)
            except Exception as exc:  # defensive: a tool must never break the loop
                output = None
                tool_error = exc
            else:
                tool_error = None
        # PostToolUse has already normalized and redacted the output before
        # it reaches the log, the ledger, or the next model request.
        post_context = HookContext(
            event="PostToolUse", session=session, session_id=session.state.session_id,
            turn=session.state.turn, workdir=workdir, tool_call_id=tool_id,
            tool_name=tool_name, arguments=arguments if argument_error is None else {}, output=output,
            error=tool_error if argument_error is None else None,
            metadata={"argument_error": True} if argument_error is not None else {},
        )
        hooks.trigger(post_context)
        _append_tool_observation(session, messages, tool_id, post_context.output or "Error: Tool execution failed")

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
    if todo_manager is not None:
        session.state.todos = [TodoItem.model_validate(item) for item in todo_manager.items]
    return used_todo


def _append_tool_observation(session: Session, messages: list[dict[str, Any]], tool_id: str, output: str) -> None:
    tool_message = {"role": "tool", "tool_call_id": tool_id, "content": output}
    session.append(tool_message)
    messages.append(tool_message)


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
        message = redact_text(message, secret_redactor(os.getenv(config.model.api_key_env, "")))
    return message[:1000]


def _task_runner(
    config: Config,
    client: Any,
    *,
    workdir: str,
    parent_session_id: str,
    permission_callback: PermissionCallback | None,
    custom_hooks: HookRegistry | None,
    console: Callable[[str], None] | None,
) -> Callable[..., str]:
    """Build the delegation entry the task tool calls from a root loop.

    Each interaction reuses ``run`` and the child's durable context. Only
    its identity, interaction status, and final text return to the parent.
    """

    manager = get_background_task_manager(
        config.runtime.sessions_dir, config.runtime.max_background_subagents
    )

    def render(
        sub: Session,
        operation_status: str,
        body: str,
        *,
        execution_status: str | None = None,
    ) -> str:
        redactor = secret_redactor(os.getenv(config.model.api_key_env, ""))
        display_name = redact_text(sub.state.name or "未命名子代理", redactor)
        return redact_text(
            f"session_id: {sub.state.session_id}\nname: {display_name}\n"
            f"status: {operation_status}\n"
            f"execution_status: {execution_status or sub.state.execution_status}\n"
            f"session_status: {sub.state.status}\n\n{body}",
            redactor,
        )

    def mark_interrupted(sub: Session) -> Session:
        if sub.state.execution_status in ("queued", "running", "cancel_requested"):
            if manager.status(sub.state.session_id) is None and sub.state.execution_owner_pid != os.getpid():
                sub.update_execution(
                    "interrupted",
                    error=ErrorInfo(code="background_interrupted", message="Background worker is no longer active."),
                    owner_pid=None,
                )
                sub = Session.open(config, sub.state.session_id)
        return sub

    def runner(
        prompt: str | None = None,
        *,
        action: str = "run",
        session_id: str | None = None,
        name: str | None = None,
        background: bool = True,
        timeout_ms: int = 0,
    ) -> str:
        redactor = secret_redactor(os.getenv(config.model.api_key_env, ""))
        try:
            if session_id is None:
                safe_name = redact_text(name or "", redactor)
                sub = Session.create(
                    config,
                    query=redact_text(prompt or "", redactor),
                    workdir=workdir,
                    system_prompt=redact_text(
                        f"{config.subagent_system_prompt(workdir)}\n\nSubagent name: {safe_name}",
                        redactor,
                    ),
                    parent_session_id=parent_session_id,
                    name=safe_name,
                )
            else:
                sub = mark_interrupted(Session.open(config, session_id))
                if sub.state.parent_session_id != parent_session_id:
                    return "Error: Subagent does not belong to this parent session"
                if normalize_workdir(sub.state.workdir) != workdir:
                    return "Error: Subagent is bound to a different workdir"
        except Exception as exc:
            return f"Error: Unable to open or create subagent: {_safe_exception_message(exc, config)}"

        if action in ("status", "wait"):
            snapshot = manager.status(sub.state.session_id)
            if action == "wait" and sub.state.execution_status in ("queued", "running", "cancel_requested"):
                snapshot = manager.wait(sub.state.session_id, timeout_ms / 1000)
                sub = mark_interrupted(Session.open(config, sub.state.session_id))
            else:
                sub = mark_interrupted(Session.open(config, sub.state.session_id))
            body = sub.state.execution_answer or (
                sub.state.execution_error.message if sub.state.execution_error else "No completed result yet."
            )
            execution_status = snapshot.status if snapshot is not None else sub.state.execution_status
            op_status = "succeeded" if execution_status == "succeeded" else (
                "running" if execution_status in ("queued", "running", "cancel_requested") else "failed"
            )
            if snapshot is not None and snapshot.error and not sub.state.execution_error:
                body = snapshot.error
            elif snapshot is not None and snapshot.answer is not None and sub.state.execution_answer is None:
                body = snapshot.answer
            return render(sub, op_status, body, execution_status=execution_status)

        if action == "close":
            snap = manager.request_cancel(sub.state.session_id)
            if snap is not None and snap.status == "cancel_requested":
                return render(
                    sub,
                    "running",
                    "Cancellation requested; the current model/tool batch will finish first.",
                    execution_status="cancel_requested",
                )
            sub = Session.open(config, sub.state.session_id)
            if snap is None and sub.state.execution_status in ("queued", "running", "cancel_requested"):
                sub = mark_interrupted(sub)
            if sub.state.execution_status == "cancelled" and sub.state.status == "running":
                sub.close()
            elif sub.state.status == "running":
                sub.close()
            sub = Session.open(config, sub.state.session_id)
            return render(sub, "succeeded", "Subagent is closed.")

        if sub.state.status != "running":
            return render(sub, "failed", "Subagent failed (session_closed): session is not running")
        if sub.state.execution_status in ("queued", "running", "cancel_requested"):
            return render(sub, "running", "Subagent already has an active background execution.")

        safe_prompt = redact_text(prompt or "", redactor)
        display_name = redact_text(sub.state.name or "未命名子代理", redactor)
        console_name = display_name.replace("\\", "\\\\").replace("|", "\\|").replace("]", "\\]")

        def execute(cancel_check: Callable[[], bool]) -> str:
            def notify(line: str) -> None:
                if console is None:
                    return
                with CONSOLE_LOCK:
                    console(f"[sub:{console_name}|{sub.state.session_id}] {line}")

            sub_console = None if console is None else (
                notify
            )
            result = run(
                sub.state.session_id,
                InputRequest(query=safe_prompt),
                config,
                openai_sdk_client=client,
                workdir=workdir,
                console=sub_console,
                hooks=custom_hooks,
                permission_callback=permission_callback,
                cancel_check=cancel_check,
            )
            if result.status != "succeeded":
                error = result.errors[0] if result.errors else None
                raise RuntimeError(f"{error.code if error else result.status}: {error.message if error else 'subagent failed'}")
            return result.answer or "(subagent returned no summary)"

        if not background:
            execution_id = uuid.uuid4().hex
            sub.update_execution("queued", execution_id=execution_id, owner_pid=os.getpid())
            sub.update_execution("running", execution_id=execution_id, owner_pid=os.getpid())
            try:
                answer = execute(lambda: False)
            except Exception as exc:
                sub.update_execution("failed", error=ErrorInfo(code="background_error", message=_safe_exception_message(exc, config)))
                sub = Session.open(config, sub.state.session_id)
                return render(sub, "failed", f"Subagent failed (background_error): {_safe_exception_message(exc, config)}")
            sub.update_execution("succeeded", answer=answer, owner_pid=os.getpid())
            return render(Session.open(config, sub.state.session_id), "succeeded", answer)

        def on_status(status: str, **metadata: Any) -> object:
            fresh = Session.open(config, sub.state.session_id)
            if status == "running":
                return fresh.claim_execution(metadata["execution_id"])
            error = metadata.get("error")
            error_info = ErrorInfo(code="background_error", message=str(error)) if error else None
            fresh.update_execution(
                status,
                execution_id=metadata["execution_id"],
                answer=metadata.get("answer"),
                error=error_info,
                owner_pid=metadata.get("owner_pid"),
            )
            if status == "cancelled" and fresh.state.status == "running":
                fresh.close()
            return None

        try:
            manager.submit(
                sub.state.session_id,
                safe_prompt,
                execute,
                parent_id=parent_session_id,
                on_status=on_status,
            )
        except ValueError as exc:
            return render(Session.open(config, sub.state.session_id), "failed", str(exc))
        sub = Session.open(config, sub.state.session_id)
        if sub.state.execution_status == "succeeded":
            return render(sub, "succeeded", sub.state.execution_answer or "(subagent returned no summary)")
        if sub.state.execution_status == "failed":
            message = sub.state.execution_error.message if sub.state.execution_error else "subagent failed"
            return render(sub, "failed", f"Subagent failed: {message}")
        return render(sub, "running", "Background subagent queued.")

    return runner


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
    # Finish the parent's locked close before acquiring any child locks.
    result = Session.open(config, session_id).close(status)  # type: ignore[arg-type]
    manager = get_background_task_manager(
        config.runtime.sessions_dir, config.runtime.max_background_subagents
    )
    failures: list[str] = []
    for state in Session.list(config):
        if state.parent_session_id != session_id:
            continue
        try:
            snapshot = manager.request_cancel(state.session_id)
            if snapshot is not None and snapshot.status in ("running", "cancel_requested"):
                manager.wait(state.session_id)
            child = Session.open(config, state.session_id)
            if snapshot is None and child.state.execution_status in ("queued", "running", "cancel_requested"):
                child.update_execution(
                    "cancelled",
                    error=ErrorInfo(
                        code="background_interrupted",
                        message="Background worker was unavailable during parent close.",
                    ),
                    owner_pid=None,
                )
                child = Session.open(config, state.session_id)
            if child.state.status == "running":
                child.close(result.status)  # type: ignore[arg-type]
        except Exception as exc:
            failures.append(f"{state.session_id}: {_safe_exception_message(exc, config)}")
    if failures:
        raise ConfigError(
            "subagent_close_failed",
            "Parent session is closed; retry close-session to finish closing subagents: "
            + "; ".join(failures),
        )
    return result


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
