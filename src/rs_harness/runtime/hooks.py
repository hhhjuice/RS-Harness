"""Lifecycle hooks, permission checks, and hook audit support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .redact import Redactor, identity, redact_text, redact_value
from .todos import is_error_observation
from .tools import BASH_DENY_LIST, safe_path

HookEvent = Literal["UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"]
PermissionCallback = Callable[["PermissionRequest"], bool]
EventSink = Callable[[dict[str, Any]], object]


@dataclass
class HookContext:
    event: HookEvent
    session: Any
    session_id: str
    turn: int
    workdir: str
    query: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: BaseException | None = None
    redact: Redactor = field(default=identity, repr=False)


@dataclass(frozen=True)
class HookDecision:
    allowed: bool = True
    reason: str | None = None
    allow_workspace_escape: bool = False


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    workdir: str


HookCallback = Callable[[HookContext], HookDecision | str | None]

# Returned by observer hooks that found nothing to do; ``trigger`` skips the
# event record so the audit ledger only carries hooks that actually acted.
_SILENT = object()

# Console prefix for the todo list notice; the CLI matches it for coloring.
TODO_NOTICE_PREFIX = "[HOOK] ## Current Tasks"


class HookRegistry:
    """Ordered callbacks keyed by lifecycle event."""

    def __init__(self, *, event_sink: EventSink | None = None, redactor: Redactor | None = None) -> None:
        self._hooks: dict[HookEvent, list[tuple[str, HookCallback]]] = {
            "UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []
        }
        self._event_sink = event_sink
        self._redactor = redactor or identity

    def register(self, event: HookEvent, callback: HookCallback, *, name: str | None = None) -> None:
        self._hooks[event].append((name or getattr(callback, "__name__", "hook"), callback))

    def callbacks(self, event: HookEvent) -> list[tuple[str, HookCallback]]:
        return list(self._hooks[event])

    def trigger(self, context: HookContext) -> HookDecision | str | None:
        """Run the callbacks for ``context.event``.

        PreToolUse returns the first denying ``HookDecision``; Stop returns the
        first non-empty string, which the loop appends as a user message to
        force continuation. Every other result is observational only.
        """

        context.redact = self._redactor
        allowed_decision: HookDecision | None = None
        for name, callback in self._hooks[context.event]:
            hook_error: BaseException | None = None
            if context.output is not None:
                context.output = context.redact(context.output)
            try:
                result = callback(context)
            except Exception as exc:
                hook_error = exc
                result = (
                    HookDecision(False, f"Hook {name} failed: {type(exc).__name__}")
                    if context.event == "PreToolUse"
                    else None
                )
            if isinstance(result, str) and context.event == "PreToolUse":
                result = HookDecision(False, result)
            if context.output is not None:
                context.output = context.redact(context.output)
            if hook_error is None and result is _SILENT:
                continue
            if result is _SILENT:
                result = None
            self._record(context, name, result, hook_error)
            if context.event == "PreToolUse" and isinstance(result, HookDecision):
                if not result.allowed:
                    return result
                allowed_decision = result
            if context.event == "Stop" and isinstance(result, str) and result.strip():
                return result
        return allowed_decision

    def _record(
        self,
        context: HookContext,
        name: str,
        result: HookDecision | str | None,
        error: BaseException | None,
    ) -> None:
        if self._event_sink is None:
            return
        denied = isinstance(result, HookDecision) and not result.allowed
        payload: dict[str, Any] = {
            "event": context.event,
            "hook": name,
            "session_id": context.session_id,
            "turn": context.turn,
            "status": "error" if error is not None else ("denied" if denied else "completed"),
        }
        if context.event == "Stop" and isinstance(context.metadata.get("status"), str):
            payload["status"] = context.metadata["status"]
        if (
            isinstance(result, str)
            and context.event == "Stop"
            and result.strip()
        ):
            payload["status"] = "force_continue"
        if context.tool_call_id is not None:
            payload["tool_call_id"] = context.tool_call_id
        if context.tool_name is not None:
            payload["tool_name"] = context.tool_name
        if context.query is not None:
            payload["query"] = context.query
        if context.arguments:
            # 与 output[:200] 同一上限：write_file 等大参数不得整体进入审计日志。
            payload["arguments"] = {
                key: value[:200] if isinstance(value, str) else value
                for key, value in context.arguments.items()
            }
        if context.output is not None:
            payload["output"] = context.output[:200]
        if context.error is not None:
            payload["error"] = context.error
        if error is not None:
            payload["hook_error"] = error
        if isinstance(result, HookDecision) and result.reason:
            payload["reason"] = result.reason
        for key, value in context.metadata.items():
            if not key.startswith("_") and key != "status":
                payload[key] = value
        try:
            self._event_sink(redact_value(payload, context.redact))
        except Exception:
            # Audit persistence must never break a model/tool interaction.
            pass


DESTRUCTIVE = ("rm ", "> /etc/", "chmod 777")


def check_permission(context: HookContext, permission_callback: PermissionCallback | None = None) -> HookDecision | None:
    """PreToolUse permission gate migrated from s03_permission."""

    name = context.tool_name or ""
    args = context.arguments
    if name == "bash":
        command = str(args.get("command", ""))
        for pattern in BASH_DENY_LIST:
            if pattern in command:
                return HookDecision(False, f"Blocked: '{pattern}' is on the deny list")
        reason = next(("Potentially destructive command" for pattern in DESTRUCTIVE if pattern in command), None)
    elif name in {"read_file", "write_file", "edit_file", "read_image"}:
        try:
            safe_path(str(args.get("path", "")), context.workdir)
        except ValueError:
            reason = "Access outside workspace"
        else:
            reason = None
    else:
        reason = None
    if reason is None:
        return None
    request = PermissionRequest(name, dict(args), reason, context.workdir)
    if permission_callback is None or not permission_callback(request):
        return HookDecision(False, f"Permission denied: {reason}")
    return HookDecision(True, reason, allow_workspace_escape=(reason == "Access outside workspace"))


def context_inject_hook(context: HookContext, console: Callable[[str], None] | None = None) -> None:
    context.metadata["working_directory"] = context.workdir
    if console is not None:
        console(f"[HOOK] UserPromptSubmit: working in {context.workdir}")


def tool_error_hook(context: HookContext) -> object:
    if context.error is None or context.output is not None:
        return _SILENT
    message = redact_text(str(context.error).strip() or type(context.error).__name__, context.redact)
    context.output = f"Error: Tool execution failed: {message[:1000]}"
    return None


def log_hook(context: HookContext, console: Callable[[str], None] | None = None) -> object:
    """PreToolUse: log every tool invocation with a short argument preview."""
    if console is None or not context.tool_name:
        return _SILENT
    preview = str(list(context.arguments.values())[:2])[:60]
    console(f"[HOOK] {context.tool_name}({redact_text(preview, context.redact)})")
    return _SILENT  # console-only; the check_permission audit already records the call


def large_output_hook(context: HookContext) -> object:
    if context.output is None or len(context.output) <= 100000:
        return _SILENT
    context.metadata["large_output"] = len(context.output)
    return None


def todo_display_hook(context: HookContext, console: Callable[[str], None] | None = None) -> object:
    """PostToolUse: surface an accepted todo_write list through the console."""
    if console is None or context.tool_name != "todo_write":
        return _SILENT
    if context.output is None or is_error_observation(context.output):
        return _SILENT
    console(f"{TODO_NOTICE_PREFIX}\n{context.output}")
    return None


def summary_hook(context: HookContext, console: Callable[[str], None] | None = None) -> None:
    messages = context.session.messages() if context.session is not None else []
    tool_count = sum(1 for message in messages if message.get("role") == "tool")
    context.metadata["tool_count"] = tool_count
    if console is not None:
        console(f"[HOOK] Stop: session used {tool_count} tool calls")


def default_hooks(
    *,
    permission_callback: PermissionCallback | None = None,
    console: Callable[[str], None] | None = None,
    event_sink: EventSink | None = None,
    redactor: Redactor | None = None,
    custom: HookRegistry | None = None,
) -> HookRegistry:
    """Build defaults and insert custom PostToolUse callbacks before observers.

    ``console`` receives one line per user-facing hook notice, each prefixed
    with its ``[HOOK]`` origin (submit, tool invocation, stop summary); the
    registry itself never prints.
    """
    registry = HookRegistry(event_sink=event_sink, redactor=redactor)
    registry.register("UserPromptSubmit", lambda ctx: context_inject_hook(ctx, console), name="context_inject_hook")
    registry.register("PreToolUse", lambda ctx: check_permission(ctx, permission_callback), name="check_permission")
    registry.register("PreToolUse", lambda ctx: log_hook(ctx, console), name="log_hook")
    registry.register("PostToolUse", tool_error_hook)
    if custom is not None:
        for event in ("UserPromptSubmit", "PreToolUse"):
            for name, callback in custom.callbacks(event):
                registry.register(event, callback, name=name)
        for name, callback in custom.callbacks("PostToolUse"):
            registry.register("PostToolUse", callback, name=name)
    registry.register("PostToolUse", large_output_hook)
    registry.register("PostToolUse", lambda ctx: todo_display_hook(ctx, console), name="todo_display_hook")
    registry.register("Stop", lambda ctx: summary_hook(ctx, console), name="summary_hook")
    if custom is not None:
        for name, callback in custom.callbacks("Stop"):
            registry.register("Stop", callback, name=name)
    return registry


__all__ = [
    "HookEvent", "HookContext", "HookDecision", "HookRegistry", "PermissionRequest",
    "PermissionCallback", "check_permission", "default_hooks", "tool_error_hook",
    "log_hook", "large_output_hook", "summary_hook", "todo_display_hook",
    "TODO_NOTICE_PREFIX",
]
