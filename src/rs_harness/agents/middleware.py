"""Official LangChain middleware used by the production Agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from inspect import isawaitable
from pathlib import Path
from typing import Literal

from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware, ModelCallLimitMiddleware, TodoListMiddleware, ToolErrorMiddleware, hook_config
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import ToolException
from openai import APIConnectionError, APITimeoutError, RateLimitError

from ..config import Config, ResolvedSubagent
from ..context_usage import CONTEXT_SOURCE
from ..inputs import InputError, validate_image_messages
from ..mcp import MCPRegistry
from ..skills import SkillCatalog
from ..tools.filesystem import resolve_file_path
from .skills import SkillMiddleware
from .mcp import MCPMiddleware


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
    elif name in {"read_file", "write_file", "edit_file", "read_image"}:
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


class PermissionMiddleware(AgentMiddleware):
    """Build one official HITL batch from local rules and the current MCP catalog."""

    def __init__(self, tools: Sequence[str], mcp: MCPRegistry | None = None):
        self.local_rules = {
            name: {"allowed_decisions": ["approve", "reject"], "when": needs_approval,
                   "description": _approval_description}
            for name in ("read_file", "write_file", "edit_file", "read_image", "bash") if name in tools
        }
        self.mcp = mcp

    def _hitl(self, state):
        rules = dict(self.local_rules)
        if self.mcp is not None:
            for name, entry in self.mcp.entries(state).items():
                rules[name] = {"allowed_decisions": ["approve", "reject"]} if self.mcp.requires_approval(entry) else False
        return HumanInTheLoopMiddleware(interrupt_on=rules)

    def after_model(self, state, runtime):
        if self.mcp is not None:
            self.mcp.check_sync(state)
        return self._hitl(state).after_model(state, runtime)

    async def aafter_model(self, state, runtime):
        if self.mcp is not None:
            await self.mcp.ensure_loaded(state)
        return await self._hitl(state).aafter_model(state, runtime)


class ToolDenyMiddleware(AgentMiddleware):
    def __init__(self, *, allow_approval: bool = True):
        self.allow_approval = allow_approval

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call.get("name")
        if name != "bash" and (self.allow_approval or name not in {"read_file", "write_file", "edit_file", "read_image"}):
            return await handler(request)
        decision, reason = classify_tool_call(request.tool_call, request.runtime.context.workdir)
        if decision == "deny" or (decision == "ask" and not self.allow_approval):
            if decision == "ask":
                reason = f"Permission denied: subagents cannot request human approval. {reason}"
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


def _delegate_error(error: Exception, request, subagents: Sequence[ResolvedSubagent]) -> str | None:
    if request.tool_call["name"] != "delegate_task":
        return _tool_error(error, request)
    if isinstance(error, ToolException):
        return str(error)
    category = next((label for kind, label in (
        (ModelCallLimitExceededError, "subagent_budget_exhausted"),
        (APITimeoutError, "subagent_model_timeout"),
        (APIConnectionError, "subagent_model_connection_error"),
        (RateLimitError, "subagent_model_rate_limit"),
    ) if isinstance(error, kind)), None)
    if category is None:
        return None
    agent_id = request.tool_call["args"].get("agent_id")
    definition = next(item for item in subagents if item.id == agent_id)
    return (
        f"{category}: {definition.name} ({definition.id}) did not finish. "
        "Earlier tools may already have changed files. Inspect existing artifacts before deciding "
        "whether to narrow the task, ask this agent again, delegate elsewhere, or report unfinished work."
    )


class SubagentBatchMiddleware(AgentMiddleware):
    """Allow one call per persistent identity in the current AIMessage batch."""

    def __init__(self, subagents: Sequence[ResolvedSubagent]):
        self.persistent_ids = {item.id for item in subagents if item.persistence == "per_thread"}

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        agent_id = call["args"].get("agent_id")
        if call["name"] == "delegate_task" and isinstance(agent_id, str) and agent_id in self.persistent_ids:
            message = next(message for message in reversed(request.state["messages"]) if isinstance(message, AIMessage))
            first = next(item for item in message.tool_calls
                         if item["name"] == "delegate_task" and item["args"].get("agent_id") == agent_id)
            if first["id"] != call["id"]:
                raise ToolException(
                    f"duplicate_persistent_subagent: {agent_id} already has a call in this batch. "
                    "Wait for its result, then ask again in the next model turn. This duplicate was not executed."
                )
        return await handler(request)


def _project_image_tool_results(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Send tool images as user content, after the complete tool batch, without changing state."""
    projected = []
    pending = []
    for message in messages:
        if not isinstance(message, ToolMessage) and pending:
            projected.append(HumanMessage(content_blocks=pending, additional_kwargs={CONTEXT_SOURCE: "tool_results"}))
            pending = []
        if (
            isinstance(message, ToolMessage) and message.name == "read_image"
            and message.status == "success" and isinstance(message.content, list)
        ):
            pending.extend(message.content)
            text = "\n".join(
                block["text"] for block in message.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            projected.append(message.model_copy(update={"content": text}))
        else:
            projected.append(message)
    if pending:
        projected.append(HumanMessage(content_blocks=pending, additional_kwargs={CONTEXT_SOURCE: "tool_results"}))
    return projected


class ImageInputMiddleware(AgentMiddleware):
    def __init__(self, config: Config):
        self.config = config

    async def awrap_model_call(self, request, handler):
        messages = _project_image_tool_results(request.messages)
        try:
            validate_image_messages(messages, self.config, getattr(request.model, "profile", None))
        except InputError as exc:
            await _notify_middleware(self.name, "error", str(exc), error_type=type(exc).__name__)
            raise
        return await handler(request.override(messages=messages))


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
                return {"messages": [HumanMessage(content=result, additional_kwargs={CONTEXT_SOURCE: "system"})], "jump_to": "model"}
            return None
        return None


def build_middleware(
    config: Config, *, tools: Sequence[str],
    subagents: Sequence[ResolvedSubagent] = (),
    skills: SkillCatalog | None = None,
    allow_approval: bool = True,
    mcp: MCPRegistry | None = None,
) -> list[AgentMiddleware]:
    """Build fresh middleware instances for one Agent graph."""

    return [
        *([PermissionMiddleware(tools, mcp)] if allow_approval else []),
        ModelCallLimitMiddleware(
            run_limit=config.runtime.max_model_calls,
            exit_behavior="error",
        ),
        ToolDenyMiddleware(allow_approval=allow_approval),
        ToolErrorMiddleware(on_error=(lambda error, request: _delegate_error(error, request, subagents)) if subagents else _tool_error),
        *([SubagentBatchMiddleware(subagents)] if subagents else []),
        ImageInputMiddleware(config),
        *([TodoListMiddleware()] if "write_todos" in tools else []),
        *([SkillMiddleware(skills)] if "load_skill" in tools and skills is not None else []),
        *([MCPMiddleware(mcp)] if mcp is not None else []),
        StopHookMiddleware(),
    ]
