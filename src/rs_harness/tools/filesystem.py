"""Local file tools with Agent-managed approval for external paths."""

from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from ..context import AgentContext
from ..inputs import load_local_image
from . import NonBlank


class PathInput(BaseModel):
    path: NonBlank = Field(description="Relative or absolute file path. Access outside the workspace requires Agent approval.")


class ReadInput(PathInput):
    offset: int = Field(default=1, gt=0, strict=True, description="First line, starting at 1.")
    limit: int = Field(default=200, gt=0, le=2000, strict=True, description="Maximum number of lines.")


class WriteInput(PathInput):
    content: str = Field(strict=True, description="Complete UTF-8 content; replaces the existing file.")


class EditInput(PathInput):
    old_text: str = Field(min_length=1, strict=True, description="Exact text that occurs once in the file.")
    new_text: str = Field(strict=True, description="Replacement text; empty text deletes the match.")


class GlobInput(BaseModel):
    pattern: NonBlank = Field(description="Workspace-relative file pattern; ** matches recursively.")


def resolve_file_path(path: str, workdir: str) -> Path:
    root = Path(workdir).resolve()
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _file_path(path: str, workdir: str) -> Path:
    resolved = resolve_file_path(path, workdir)
    if resolved.exists() and not resolved.is_file():
        raise ToolException(f"Not a regular file: {path}")
    return resolved


def _display_path(target: Path, workdir: str) -> str:
    root = Path(workdir).resolve()
    return str(target.relative_to(root) if target.is_relative_to(root) else target)


@tool(args_schema=ReadInput)
def read_file(path: str, runtime: ToolRuntime[AgentContext], offset: int = 1, limit: int = 200) -> dict:
    """Read UTF-8 text with line and output limits. External paths require Agent approval."""
    try:
        target = _file_path(path, runtime.context.workdir)
        display_path = _display_path(target, runtime.context.workdir)
        remaining = runtime.context.max_tool_output_bytes
        parts = []
        end_line = None
        truncated = False
        with target.open(encoding="utf-8", newline="") as handle:
            skipped = 0
            while skipped < offset - 1:
                chunk = handle.readline(8192)
                if not chunk:
                    break
                if chunk.endswith("\r"):
                    position = handle.tell()
                    if handle.read(1) != "\n":
                        handle.seek(position)
                if chunk.endswith(("\n", "\r")):
                    skipped += 1
            for line_number in range(offset, offset + limit):
                line = handle.readline(remaining + 1)
                if not line:
                    break
                if "\x00" in line:
                    raise ToolException(f"Not a UTF-8 text file: {path}")
                encoded = line.encode("utf-8")
                if len(encoded) > remaining:
                    fragment = encoded[:remaining].decode("utf-8", errors="ignore")
                    parts.append(fragment)
                    if fragment:
                        end_line = line_number
                    truncated = True
                    break
                parts.append(line)
                end_line = line_number
                remaining -= len(encoded)
            else:
                truncated = bool(handle.read(1))
        return {
            "path": display_path,
            "content": "".join(parts),
            "start_line": offset,
            "end_line": end_line,
            "truncated": truncated,
        }
    except (OSError, UnicodeError, ValueError) as exc:
        raise ToolException(f"read_file: {exc}") from exc


@tool(args_schema=PathInput)
def read_image(path: str, runtime: ToolRuntime[AgentContext]) -> list[dict]:
    """View a local JPEG, PNG, GIF, or WebP image up to 32 MiB. External paths require Agent approval."""
    try:
        target = _file_path(path, runtime.context.workdir)
        block = load_local_image(target)
        return [
            {"type": "text", "text": f"Image loaded from {_display_path(target, runtime.context.workdir)}"},
            block,
        ]
    except (OSError, ValueError) as exc:
        raise ToolException(f"read_image: {exc}") from exc


@tool(args_schema=WriteInput)
def write_file(path: str, content: str, runtime: ToolRuntime[AgentContext]) -> dict:
    """Create or overwrite a file with an existing parent. External paths require Agent approval."""
    try:
        target = _file_path(path, runtime.context.workdir)
        display_path = _display_path(target, runtime.context.workdir)
        written = target.write_bytes(content.encode("utf-8"))
        return {"path": display_path, "bytes_written": written}
    except (OSError, UnicodeError, ValueError) as exc:
        raise ToolException(f"write_file: {exc}") from exc


@tool(args_schema=EditInput)
def edit_file(path: str, old_text: str, new_text: str, runtime: ToolRuntime[AgentContext]) -> dict:
    """Replace one unique exact text match in a file. External paths require Agent approval."""
    try:
        target = _file_path(path, runtime.context.workdir)
        display_path = _display_path(target, runtime.context.workdir)
        content = target.read_bytes().decode("utf-8")
        if "\x00" in content:
            raise ToolException(f"Not a UTF-8 text file: {path}")
        if not old_text or content.count(old_text) != 1:
            raise ToolException("old_text must occur exactly once; read the file and provide a unique match")
        target.write_bytes(content.replace(old_text, new_text, 1).encode("utf-8"))
        return {"path": display_path, "replacements": 1}
    except (OSError, UnicodeError, ValueError) as exc:
        raise ToolException(f"edit_file: {exc}") from exc


@tool(args_schema=GlobInput)
def glob(pattern: str, runtime: ToolRuntime[AgentContext]) -> dict:
    """Find workspace files, including dotfiles. Returns sorted paths, capped at 200 and the output budget."""
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ToolException("glob requires a relative pattern without parent-directory segments")
    try:
        root = Path(runtime.context.workdir).resolve()
        paths = sorted(
            str(candidate.relative_to(root))
            for candidate in root.glob(pattern)
            if candidate.resolve().is_relative_to(root) and candidate.is_file()
        )
        selected = []
        remaining = runtime.context.max_tool_output_bytes
        for path in paths[:200]:
            size = len(path.encode("utf-8")) + 1
            if size > remaining:
                break
            selected.append(path)
            remaining -= size
        return {"paths": selected, "truncated": len(selected) < len(paths)}
    except (OSError, ValueError) as exc:
        raise ToolException(f"glob: {exc}") from exc
