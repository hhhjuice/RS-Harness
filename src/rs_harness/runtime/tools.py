"""Built-in workspace tools used by the model loop."""

from __future__ import annotations

import glob as glob_module
import os
import signal
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .images import ImageAttachment, inspect_local_image
from .todos import TodoManager, run_todo_write

__all__ = [
    "TOOLS",
    "SUB_TOOLS",
    "TOOL_HANDLERS",
    "dispatch_tool",
    "normalize_workdir",
    "safe_path",
    "run_bash",
    "run_read",
    "run_read_image",
    "run_write",
    "run_edit",
    "run_glob",
    "run_todo_write",
    "run_task",
]

_MAX_OUTPUT = 50000

# Hard-deny shell patterns shared by the PreToolUse permission gate and the
# run_bash inner fence; both sides read this single list.
BASH_DENY_LIST = ("rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/")

# Delegation prompts travel whole into the subagent context.
_MAX_TASK_PROMPT = 8000


def _truncate(value: str) -> str:
    return value[:_MAX_OUTPUT]


def normalize_workdir(workdir: str | Path | None = None) -> str:
    """Resolve and validate a tool workspace, returning an absolute path."""

    root = Path.cwd() if workdir is None else Path(workdir).expanduser()
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Workdir is not a directory: {workdir or root}")
    return str(root)


def _workspace(workdir: str | Path | None = None) -> Path:
    return Path(normalize_workdir(workdir))


def safe_path(
    path: str | Path,
    workdir: str | Path | None = None,
    *,
    allow_workspace_escape: bool = False,
) -> Path:
    """Resolve a path, fencing it to the workspace unless explicitly allowed."""

    if not isinstance(path, (str, Path)):
        raise ValueError("path must be a string")
    root = _workspace(workdir)
    candidate = (root / path).resolve()
    if not allow_workspace_escape and not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes workspace: {path}")
    return candidate


def run_bash(
    command: str, workdir: str | Path | None = None, timeout_s: float = 120
) -> str:
    # Keep shell execution aligned with s02: select the workspace as cwd and
    # apply its command blocklist; file-tool path fencing is handled separately.
    try:
        if not isinstance(command, str):
            return "Error: Invalid command: expected a string"
        dangerous = BASH_DENY_LIST
        if any(item in command for item in dangerous):
            return "Error: Dangerous command blocked"
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=_workspace(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # start_new_session makes the shell a process-group leader, so the
            # kill reaches the whole tree instead of orphaning grandchildren.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.communicate(timeout=5)
            except Exception:
                pass
            return f"Error: Timeout ({timeout_s:g}s)"
        output = ((stdout or "") + (stderr or "")).strip()
        return _truncate(output) if output else "(no output)"
    except Exception as exc:
        return _truncate(f"Error: {exc}")


def run_read(
    path: str | Path,
    limit: int | None = None,
    workdir: str | Path | None = None,
    *,
    allow_workspace_escape: bool = False,
) -> str:
    try:
        if not isinstance(path, (str, Path)):
            return "Error: Invalid path: expected a string"
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            return "Error: limit must be a positive integer"
        lines = safe_path(path, workdir, allow_workspace_escape=allow_workspace_escape).read_text(encoding="utf-8").splitlines()
        if limit is not None:
            if limit < len(lines):
                lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return _truncate("\n".join(lines))
    except Exception as exc:
        return _truncate(f"Error: {exc}")


def run_read_image(
    path: str | Path,
    workdir: str | Path | None = None,
    *,
    allow_workspace_escape: bool = False,
) -> ImageAttachment | str:
    """Load one local web image for the model's next visual context."""

    try:
        if not isinstance(path, (str, Path)):
            return "Error: Invalid path: expected a string"
        target = safe_path(path, workdir, allow_workspace_escape=allow_workspace_escape)
        attachment, reason = inspect_local_image(target)
        if attachment is None:
            return _truncate(f"Error: Unable to read image {path}: {reason}")
        return attachment
    except Exception as exc:
        return _truncate(f"Error: {exc}")


def run_write(
    path: str | Path,
    content: str,
    workdir: str | Path | None = None,
    *,
    allow_workspace_escape: bool = False,
) -> str:
    try:
        if not isinstance(path, (str, Path)) or not isinstance(content, str):
            return "Error: Invalid arguments: path and content must be strings"
        target = safe_path(path, workdir, allow_workspace_escape=allow_workspace_escape)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return _truncate(f"Wrote {len(content.encode('utf-8'))} bytes to {path}")
    except Exception as exc:
        return _truncate(f"Error: {exc}")


def run_edit(
    path: str | Path,
    old_text: str,
    new_text: str,
    workdir: str | Path | None = None,
    *,
    allow_workspace_escape: bool = False,
) -> str:
    try:
        if not isinstance(path, (str, Path)) or not isinstance(old_text, str) or not isinstance(new_text, str):
            return "Error: Invalid arguments: path, old_text, and new_text must be strings"
        target = safe_path(path, workdir, allow_workspace_escape=allow_workspace_escape)
        content = target.read_text(encoding="utf-8")
        if old_text not in content:
            return _truncate(f"Error: Text not found in {path}")
        target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return _truncate(f"Edited {path}")
    except Exception as exc:
        return _truncate(f"Error: {exc}")


def run_glob(pattern: str, workdir: str | Path | None = None) -> str:
    """Return workspace-relative paths matching a glob pattern."""

    try:
        if not isinstance(pattern, str):
            return "Error: Invalid pattern: expected a string"
        root = _workspace(workdir)
        matches = sorted(
            match
            for match in glob_module.glob(pattern, root_dir=root)
            if (root / match).resolve().is_relative_to(root)
        )
        return _truncate("\n".join(matches)) if matches else "(no matches)"
    except Exception as exc:
        return _truncate(f"Error: {exc}")


def run_task(
    prompt: str | None = None,
    runner: Callable[..., str] | None = None,
    *,
    action: str = "run",
    session_id: str | None = None,
    name: str | None = None,
    background: bool = True,
    timeout_ms: int = 0,
) -> str:
    """Create, continue, or close a named subagent conversation.

    ``runner`` is injected by the parent agent loop; it owns the client,
    config, hooks, and the parent session id. Without it (direct dispatch or
    a subagent's own model hallucinating a call) delegation is refused.
    """

    if action not in ("run", "status", "wait", "close"):
        return "Error: Invalid action: expected run, status, wait, or close"
    if not isinstance(background, bool):
        return "Error: Invalid background: expected a boolean"
    if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms < 0:
        return "Error: Invalid timeout_ms: expected a non-negative integer"
    if session_id is not None and (
        not isinstance(session_id, str)
        or not session_id
        or session_id in (".", "..")
        or Path(session_id).name != session_id
        or any(char.isspace() or not char.isprintable() for char in session_id)
    ):
        return "Error: Invalid session_id: expected a non-empty session identifier"
    if action in ("status", "wait", "close"):
        if session_id is None:
            return f"Error: {action} requires session_id"
        if prompt is not None or name is not None:
            return f"Error: {action} does not accept prompt or name"
    else:
        if not isinstance(prompt, str) or not prompt.strip():
            return "Error: Invalid prompt: expected a non-empty string"
        if len(prompt) > _MAX_TASK_PROMPT:
            return _truncate(
                f"Error: Prompt too long ({len(prompt)} chars, max {_MAX_TASK_PROMPT})"
            )
        if session_id is None:
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 64:
                return "Error: Creating a subagent requires name with 1-64 characters"
            if not name.isprintable():
                return "Error: Invalid name: newlines and control characters are not allowed"
            name = name.strip()
        elif name is not None:
            return "Error: Continuing a subagent does not accept name"
    if runner is None:
        return "Error: task tool requires an active parent agent loop"
    try:
        return _truncate(str(runner(
            prompt,
            action=action,
            session_id=session_id,
            name=name,
            background=background,
            timeout_ms=timeout_ms,
        )))
    except Exception as exc:
        return _truncate(f"Error: {exc}")


# Single source of truth: each tool's schema and handler in one place.
_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "bash": {
        "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}},
        "handler": run_bash,
    },
    "read_file": {
        "function": {"name": "read_file", "description": "Read file contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}}, "required": ["path"], "additionalProperties": False}},
        "handler": run_read,
    },
    "write_file": {
        "function": {"name": "write_file", "description": "Write content to a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False}},
        "handler": run_write,
    },
    "edit_file": {
        "function": {"name": "edit_file", "description": "Replace exact text in a file once.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False}},
        "handler": run_edit,
    },
    "glob": {
        "function": {"name": "glob", "description": "Find files matching a glob pattern.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"], "additionalProperties": False}},
        "handler": run_glob,
    },
    "read_image": {
        "function": {"name": "read_image", "description": "Load a local JPEG, PNG, WebP, or GIF so the model can inspect its visible content.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}},
        "handler": run_read_image,
    },
    "todo_write": {
        "function": {"name": "todo_write", "description": "Create and manage a task list for your current coding session.", "parameters": {"type": "object", "properties": {"todos": {"type": "array", "maxItems": 20, "items": {"type": "object", "properties": {"content": {"type": "string", "minLength": 1}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"], "additionalProperties": False}}}, "required": ["todos"], "additionalProperties": False}},
        "handler": run_todo_write,
    },
    "task": {
        "function": {"name": "task", "description": "Create or continue a named subagent (background by default), query it with status or wait, or close it with action=close. Use the returned session_id for later interactions. Do not supply name when continuing, querying, or closing; close accepts no prompt.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["run", "status", "wait", "close"], "default": "run"}, "session_id": {"type": "string", "minLength": 1}, "name": {"type": "string", "minLength": 1, "maxLength": 64}, "prompt": {"type": "string", "minLength": 1, "maxLength": _MAX_TASK_PROMPT}, "background": {"type": "boolean", "default": True}, "timeout_ms": {"type": "integer", "minimum": 0, "default": 0}}, "required": [], "additionalProperties": False}},
        "handler": run_task,
    },
}

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": spec["function"]} for spec in _TOOL_SPECS.values()
]
# The subagent sees every tool except task itself: delegation is one level.
SUB_TOOLS: list[dict[str, Any]] = [
    spec for spec in TOOLS if spec["function"]["name"] != "task"
]
TOOL_HANDLERS = {name: spec["handler"] for name, spec in _TOOL_SPECS.items()}


def dispatch_tool(
    name: str,
    arguments: Mapping[str, Any],
    workdir: str | Path | None = None,
    *,
    bash_timeout_s: float | None = None,
    allow_workspace_escape: bool = False,
    todo_manager: TodoManager | None = None,
    task_runner: Callable[..., str] | None = None,
) -> str | ImageAttachment:
    """Dispatch one decoded call and turn every handler failure into text."""

    spec = _TOOL_SPECS.get(name) if isinstance(name, str) else None
    if spec is None:
        return _truncate(f"Unknown tool: {name}")
    if not isinstance(arguments, Mapping):
        return _truncate(f"Error: Invalid arguments for {name}: expected an object")
    parameters = spec["function"]["parameters"]
    try:
        values = dict(arguments)
        missing = set(parameters["required"]) - values.keys()
        if missing:
            return _truncate(
                f"Error: Invalid arguments for {name}: missing {', '.join(sorted(missing))}"
            )
        extra = values.keys() - parameters["properties"].keys()
        if extra:
            return _truncate(
                f"Error: Invalid arguments for {name}: unexpected {', '.join(sorted(extra))}"
            )
        # todo_write and task carry their own loop context instead of workdir.
        if name not in ("todo_write", "task"):
            values["workdir"] = workdir
        if name in {"read_file", "write_file", "edit_file", "read_image"}:
            values["allow_workspace_escape"] = allow_workspace_escape
        if name == "bash" and bash_timeout_s is not None:
            values["timeout_s"] = bash_timeout_s
        if name == "todo_write":
            values["manager"] = todo_manager
        if name == "task":
            values["runner"] = task_runner
        result = spec["handler"](**values)
        if isinstance(result, ImageAttachment):
            return result
        return _truncate(str(result))
    except Exception as exc:
        return _truncate(f"Error: Invalid arguments for {name}: {exc}")
