import sys
from pathlib import Path

from rs_harness.runtime.tools import (
    TOOLS,
    dispatch_tool,
    run_bash,
    run_edit,
    run_glob,
    run_read,
    run_write,
    safe_path,
)


def test_tool_definitions_expose_all_workspace_tools():
    names = [item["function"]["name"] for item in TOOLS]
    assert names == ["bash", "read_file", "write_file", "edit_file", "glob"]
    assert all(item["type"] == "function" for item in TOOLS)


def test_file_tools_write_read_and_edit_with_workspace_boundary(tmp_path: Path):
    assert run_write("nested/note.txt", "one\ntwo\nthree", tmp_path).startswith("Wrote")
    assert run_read("nested/note.txt", 2, tmp_path) == "one\ntwo\n... (1 more lines)"
    assert run_edit("nested/note.txt", "two", "TWO", tmp_path) == "Edited nested/note.txt"
    assert (tmp_path / "nested/note.txt").read_text(encoding="utf-8") == "one\nTWO\nthree"
    assert run_edit("nested/note.txt", "", "start-", tmp_path) == "Edited nested/note.txt"
    assert (tmp_path / "nested/note.txt").read_text(encoding="utf-8").startswith("start-")
    assert "Path escapes workspace" in run_read("../outside.txt", workdir=tmp_path)
    assert safe_path("nested/note.txt", tmp_path) == (tmp_path / "nested/note.txt").resolve()


def test_run_write_reports_utf8_bytes_not_characters(tmp_path: Path):
    assert run_write("unicode.txt", "héllo", tmp_path) == "Wrote 6 bytes to unicode.txt"


def test_file_tool_errors_are_strings_and_symlink_escape_is_rejected(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-rsh-tool-test.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    assert "Path escapes workspace" in run_read("link.txt", workdir=tmp_path)
    assert run_edit("missing.txt", "old", "new", tmp_path).startswith("Error:")
    assert "positive integer" in run_read("link.txt", 0, tmp_path)
    assert dispatch_tool("unknown", {}, tmp_path) == "Unknown tool: unknown"
    assert "missing" in dispatch_tool("write_file", {"path": "x"}, tmp_path)
    assert "unexpected" in dispatch_tool("bash", {"command": "true", "extra": 1}, tmp_path)
    outside.unlink()


def test_bash_runs_in_workspace_and_truncates_output(tmp_path: Path):
    assert run_bash("pwd", tmp_path) == str(tmp_path.resolve())
    assert run_bash("sudo echo blocked", tmp_path) == "Error: Dangerous command blocked"
    output = run_bash(f"\"{sys.executable}\" -c \"print('x' * 60000)\"", tmp_path)
    assert len(output) == 50000


def test_bash_timeout_returns_error_after_deadline(tmp_path: Path):
    assert run_bash("sleep 30", tmp_path, timeout_s=0.5) == "Error: Timeout (0.5s)"


def test_glob_returns_workspace_relative_matches_and_filters_escape(tmp_path: Path):
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.py").write_text("b", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-glob-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "outside-link.txt").symlink_to(outside)

    assert run_glob("*.py", workdir=tmp_path) == "a.py\nb.py"
    assert run_glob("nested/*.py", workdir=tmp_path) == "nested/b.py"
    assert run_glob("*.txt", workdir=tmp_path) == "(no matches)"
    assert run_glob(str(outside), workdir=tmp_path) == "(no matches)"
    outside.unlink()


def test_glob_invalid_pattern_argument_is_a_string_error(tmp_path: Path):
    assert run_glob(123, workdir=tmp_path).startswith("Error:")
