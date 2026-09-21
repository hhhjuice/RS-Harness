"""Concise runtime activity for the human-facing CLI."""

from __future__ import annotations

import json
import re
import shlex
import sys
import unicodedata
from asyncio import CancelledError
from time import monotonic
from typing import Any
from uuid import UUID

import typer
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from pydantic import ValidationError


_HOOKS = {"before_agent", "before_model", "after_model", "after_agent"}
_SECRET = re.compile(r"(?i)(api[_-]?key|access[_-]?token|password|secret|authorization)")
_COMMAND_ASSIGNMENT = re.compile(r"(?<![\w-])((?:--?)?[A-Za-z_][A-Za-z0-9_-]*)\s*=")
_AUTHORIZATION = re.compile(r"(?i)(?:-H|--header=)?authorization\s*:")
_COMMAND_PART = re.compile(
    r"""[\s;&|()<>]+|(?:[^\s;&|()<>'"\\]+|\\[\s\S]|'[^']*'|"(?:[^"\\]|\\[\s\S])*")+"""
)


def _redact_command(command: str) -> str:
    """Redact shell words without reformatting the remaining command."""
    parts = []
    position = 0
    while position < len(command):
        match = _COMMAND_PART.match(command, position)
        if match is None:
            return "[redacted command]"
        raw = match.group()
        position = match.end()
        try:
            words = shlex.split(raw.replace("\\\n", ""))
        except ValueError:
            return "[redacted command]"
        word = words[0] if len(words) == 1 else ""
        assignments = [item for item in _COMMAND_ASSIGNMENT.finditer(word) if _SECRET.search(item[1])]
        header = _AUTHORIZATION.search(word)
        if assignments or header:
            # Nested shell evaluation and heredocs need a full parser to bound safely.
            if any(marker in command for marker in ("$(", "`", "<<")):
                return "[redacted command]"
            if header and header.start() == 0:
                raw = shlex.quote(word[:header.end()] + " [redacted]")
            elif assignments and assignments[0].start() == 0:
                raw = assignments[0][1] + "=[redacted]"
            else:
                return "[redacted command]"
        parts.append(raw)
    return "".join(parts)


def _preview(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    escapes = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    text = "".join(
        escapes.get(character, f"\\u{ord(character):04x}")
        if unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"} else character
        for character in text
    )
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _arguments(inputs: Any) -> str:
    if not isinstance(inputs, dict):
        return ""
    summary = {}
    for key, value in inputs.items():
        if _SECRET.search(key):
            summary[key] = "[redacted]"
        elif key in {"content", "old_text", "new_text"} and isinstance(value, str):
            summary[key] = f"<{len(value)} 字符>"
        elif key in {"path", "pattern", "command", "offset", "limit"}:
            if key == "command" and isinstance(value, str):
                value = _redact_command(value)
            summary[key] = value
    return _preview(summary) if summary else ""


class CliProgress(AsyncCallbackHandler):
    raise_error = False
    run_inline = True

    def __init__(self) -> None:
        self._tools: dict[UUID, tuple[str, float, str]] = {}
        self._middleware: dict[UUID, str] = {}
        self._model_count = 0
        self._approval_counts: tuple[int, int] | None = None
        self._reported_errors: set[tuple[str, str]] = set()

    def _write(self, message: str, *, plan: bool = False) -> bool:
        try:
            color = getattr(sys.stderr, "isatty", lambda: False)()
            typer.secho(message, file=sys.stderr, fg="yellow" if plan else "bright_black", color=color)
        except (OSError, ValueError):
            return False
        return True

    def has_reported_error(self, error: dict[str, str] | None) -> bool:
        return bool(error and (error["type"], error["message"]) in self._reported_errors)

    def _report_error(self, prefix: str, kind: str, message: str, *, detail: str | None = None, plan: bool = False) -> None:
        if self._write(f"{prefix} ({_preview(kind)}): {_preview(message if detail is None else detail)}", plan=plan):
            self._reported_errors.add((kind, message))

    async def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        self._model_count += 1
        self._write(f"[模型] 第 {self._model_count} 次调用")

    async def on_chain_start(self, serialized, inputs, *, run_id: UUID, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name", "")
        if name == "rsh_agent" and isinstance(inputs, Command) and isinstance(inputs.resume, dict):
            decisions = [decision["type"] for batch in inputs.resume.values() for decision in batch["decisions"]]
            self._approval_counts = (decisions.count("approve"), decisions.count("reject"))
        _, separator, hook = name.rpartition(".")
        if separator and hook in _HOOKS:
            self._middleware[run_id] = name

    async def on_chain_end(self, outputs, *, run_id: UUID, **kwargs: Any) -> None:
        name = self._middleware.pop(run_id, None)
        if not name or not isinstance(outputs, dict):
            return
        if name == "StopHookMiddleware.after_agent" and outputs.get("jump_to") == "model":
            self._write(f"[中间件] {_preview(name)} · 继续调用模型")
        elif name == "HumanInTheLoopMiddleware.after_model" and outputs.get("messages") and self._approval_counts is not None:
            approved, rejected = self._approval_counts
            self._approval_counts = None
            self._write(f"[审批] 审批结果：批准 {approved} 项，拒绝 {rejected} 项")
        elif name == "TodoListMiddleware.after_model":
            rejected = [message for message in outputs.get("messages", []) if isinstance(message, ToolMessage) and message.status == "error"]
            if rejected:
                self._write(f"[计划] 更新被拒绝：同一响应只能更新一次（{len(rejected)} 次调用），计划未改变", plan=True)

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        name = self._middleware.pop(run_id, None)
        if name and not isinstance(error, (GraphBubbleUp, CancelledError)):
            detail = "模型调用预算已耗尽，停止继续调用" if type(error).__name__ == "ModelCallLimitExceededError" else None
            self._report_error(f"[中间件] {_preview(name)} · 失败", type(error).__name__, str(error), detail=detail)

    async def on_custom_event(self, name: str, data: Any, **kwargs: Any) -> None:
        if name != "rsh.middleware" or not isinstance(data, dict):
            return
        label = f"[中间件] {_preview(data.get('name', 'unknown'))}"
        if data.get("status") == "blocked":
            self._write(f"{label} · 已拦截 {_preview(data.get('detail', ''))}")
        elif data.get("status") == "error":
            self._report_error(f"{label} · 检查失败", data["error_type"], data["detail"])

    async def on_tool_start(
        self, serialized, input_str, *, run_id: UUID,
        inputs: dict[str, Any] | None = None, **kwargs: Any,
    ) -> None:
        name = str((serialized or {}).get("name") or "unknown")
        summary = _arguments(inputs)
        self._tools[run_id] = (name, monotonic(), summary)
        if name != "write_todos":
            self._write(f"[工具] {_preview(name)} · 开始 {summary}".rstrip())

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._tools.pop(run_id, None)
        if run is None:
            return
        name, started, summary = run
        if getattr(output, "status", None) == "error":
            prefix = "[计划] 更新失败" if name == "write_todos" else f"[工具] {_preview(name)} · 失败 ({monotonic() - started:.2f}s) {summary}".rstrip()
            self._report_error(prefix, "ToolError", str(output.content), plan=name == "write_todos")
            return
        if name != "write_todos":
            self._write(f"[工具] {_preview(name)} · 完成 ({monotonic() - started:.2f}s) {summary}".rstrip())
        update = getattr(output, "update", None)
        if name == "write_todos" and isinstance(update, dict) and isinstance(update.get("todos"), list):
            todos = update["todos"]
            self._write(f"[计划] 已更新 {len(todos)} 项" if todos else "[计划] 计划已清空", plan=True)
            for todo in todos[:3]:
                self._write(f"  {_preview(todo.get('status', 'pending'), 24)}: {_preview(todo.get('content', ''), 100)}", plan=True)
            if len(todos) > 3:
                self._write(f"  … 其余 {len(todos) - 3} 项", plan=True)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._tools.pop(run_id, None)
        if run is not None and not isinstance(error, (GraphBubbleUp, CancelledError)):
            name, started, summary = run
            prefix = "[计划] 更新失败" if name == "write_todos" else f"[工具] {_preview(name)} · 失败 ({monotonic() - started:.2f}s) {summary}".rstrip()
            detail = None
            if isinstance(error, ValidationError):
                issue = error.errors(include_input=False, include_url=False)[0]
                detail = f"{'.'.join(map(str, issue['loc']))}: {issue['msg']}"
            self._report_error(prefix, type(error).__name__, str(error), detail=detail, plan=name == "write_todos")
