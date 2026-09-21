"""Official LangChain middleware used by the production Agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from inspect import isawaitable
from pathlib import Path
from typing import Literal

from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware, ModelCallLimitMiddleware, TodoListMiddleware, ToolErrorMiddleware, hook_config
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import ToolException

from ..config import Config
from ..inputs import InputError, validate_image_messages
from ..tools.filesystem import resolve_file_path


def classify_tool_call(tool_call: dict, workdir: str) -> tuple[Literal["deny", "ask", "allow"], str]:
    name = tool_call.get("name")
    arguments = tool_call.get("args", {})
    if not isinstance(arguments, dict):
        return "allow", ""
    if name == "bash":
        command = arguments.get("command")
        if isinstance(command, str):
            for pattern in ("rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"):
                if pattern in command:
                    return "deny", f"Permission denied: Bash contains forbidden substring {pattern!r}"
            for pattern in ("rm ", "> /etc/", "chmod 777"):
                if pattern in command:
                    return "ask", f"Bash requires approval: contains dangerous substring {pattern!r}"
    elif name in {"read_file", "write_file", "edit_file"}:
        path = arguments.get("path")
        if isinstance(path, str) and path.strip():
            try:
                target = resolve_file_path(path, workdir)
            except (OSError, ValueError, RuntimeError):
                return "allow", ""
            if not target.is_relative_to(Path(workdir).resolve()):
                return "ask", f"Access outside workspace requires approval. Resolved path: {target}"
    return "allow", ""


def needs_approval(request) -> bool:
    return classify_tool_call(request.tool_call, request.runtime.context.workdir)[0] == "ask"


def _approval_description(tool_call, state, runtime) -> str:
    return classify_tool_call(tool_call, runtime.context.workdir)[1]


class ToolDenyMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        if request.tool_call.get("name") == "bash":
            decision, reason = classify_tool_call(request.tool_call, request.runtime.context.workdir)
            if decision == "deny":
                await _notify_middleware(self.name, "blocked", f"{request.tool_call['name']}: {reason}")
                return ToolMessage(
                    content=reason, status="error", name=request.tool_call["name"],
                    tool_call_id=request.tool_call["id"],
                )
        return await handler(request)


def _tool_error(error: Exception, request) -> str | None:
    if isinstance(error, ToolException):
        return str(error)
    return None


class ImageInputMiddleware(AgentMiddleware):
    def __init__(self, config: Config):
        self.config = config

    async def awrap_model_call(self, request, handler):
        try:
            validate_image_messages(request.messages, self.config, getattr(request.model, "profile", None))
        except InputError as exc:
            await _notify_middleware(self.name, "error", str(exc), error_type=type(exc).__name__)
            raise
        return await handler(request)


async def _notify_middleware(name: str, status: str, detail: str, *, error_type: str | None = None) -> None:
    config = ensure_config()
    if config.get("callbacks"):
        await adispatch_custom_event(
            "rsh.middleware", {"name": name, "status": status, "detail": detail, "error_type": error_type}, config=config,
        )


StopHook = Callable[[list[BaseMessage]], str | None | Awaitable[str | None]]


class StopHookMiddleware(AgentMiddleware):
    def __init__(self, hooks: Sequence[StopHook] = ()):
        self.hooks = tuple(hooks)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state, runtime):
        messages = state["messages"]
        if not messages or not isinstance(messages[-1], AIMessage) or messages[-1].tool_calls:
            return None
        for hook in self.hooks:
            result = hook(messages)
            if isawaitable(result):
                result = await result
            if result is None:
                continue
            if not isinstance(result, str):
                raise TypeError("Stop hook must return str or None")
            if result:
                return {"messages": [HumanMessage(content=result)], "jump_to": "model"}
            return None
        return None


def build_middleware(config: Config) -> list[AgentMiddleware]:
    """Build fresh middleware instances for one Agent graph."""

    return [
        HumanInTheLoopMiddleware(interrupt_on={
            **{
                name: {
                    "allowed_decisions": ["approve", "reject"],
                    "when": needs_approval,
                    "description": _approval_description,
                }
                for name in ("read_file", "write_file", "edit_file", "bash")
            },
            "glob": False,
        }),
        ModelCallLimitMiddleware(
            run_limit=config.runtime.max_model_calls,
            exit_behavior="error",
        ),
        ToolDenyMiddleware(),
        ToolErrorMiddleware(on_error=_tool_error),
        ImageInputMiddleware(config),
        TodoListMiddleware(),
        StopHookMiddleware(),
    ]
