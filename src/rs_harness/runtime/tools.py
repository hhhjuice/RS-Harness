"""Built-in workspace tools used by the model loop."""

from __future__ import annotations

import glob as glob_module
import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "TOOLS",
    "TOOL_HANDLERS",
    "dispatch_tool",
    "normalize_workdir",
    "safe_path",
    "run_bash",
    "run_read",
    "run_write",
    "run_edit",
    "run_glob",
]

_MAX_OUTPUT = 50000


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


def safe_path(path: str | Path, workdir: str | Path | None = None) -> Path:
    """Resolve a path under the workspace and reject traversal/symlink escape."""

    if not isinstance(path, (str, Path)):
        raise ValueError("path must be a string")
    root = _workspace(workdir)
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
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
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
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


def run_read(path: str | Path, limit: int | None = None, workdir: str | Path | None = None) -> str:
    try:
        if not isinstance(path, (str, Path)):
            return "Error: Invalid path: expected a string"
        # Accept both ``run_read(path, limit, workdir)`` and the convenient
        # ``run_read(path, workdir)`` form for embedders.
        if isinstance(limit, (str, Path)):
            if workdir is None:
                workdir, limit = limit, None
            elif isinstance(workdir, int) and not isinstance(workdir, bool):
                workdir, limit = limit, workdir
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            return "Error: limit must be a positive integer"
        lines = safe_path(path, workdir).read_text(encoding="utf-8").splitlines()
        if limit is not None:
            if limit < len(lines):
                lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return _truncate("\n".join(lines))
    except Exception as exc:
        return _truncate(f"Error: {exc}")


def run_write(path: str | Path, content: str, workdir: str | Path | None = None) -> str:
    try:
        if not isinstance(path, (str, Path)) or not isinstance(content, str):
            return "Error: Invalid arguments: path and content must be strings"
        target = safe_path(path, workdir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return _truncate(f"Wrote {len(content.encode('utf-8'))} bytes to {path}")
    except Exception as exc:
        return _truncate(f"Error: {exc}")


def run_edit(path: str | Path, old_text: str, new_text: str, workdir: str | Path | None = None) -> str:
    try:
        if not isinstance(path, (str, Path)) or not isinstance(old_text, str) or not isinstance(new_text, str):
            return "Error: Invalid arguments: path, old_text, and new_text must be strings"
        target = safe_path(path, workdir)
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
}

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": spec["function"]} for spec in _TOOL_SPECS.values()
]
TOOL_HANDLERS = {name: spec["handler"] for name, spec in _TOOL_SPECS.items()}


def dispatch_tool(
    name: str,
    arguments: Mapping[str, Any],
    workdir: str | Path | None = None,
    *,
    bash_timeout_s: float | None = None,
) -> str:
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
        values["workdir"] = workdir
        if name == "bash" and bash_timeout_s is not None:
            values["timeout_s"] = bash_timeout_s
        return _truncate(str(spec["handler"](**values)))
    except Exception as exc:
        return _truncate(f"Error: Invalid arguments for {name}: {exc}")
