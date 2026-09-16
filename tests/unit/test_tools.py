import sys
from pathlib import Path

import pytest

from rs_harness.runtime.tools import (
    SUB_TOOLS,
    TOOLS,
    dispatch_tool,
    run_bash,
    run_edit,
    run_glob,
    run_read,
    run_read_image,
    run_task,
    run_write,
    safe_path,
)
from rs_harness.runtime.images import ImageAttachment

JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-payload"


def test_tool_definitions_expose_all_workspace_tools():
    names = [item["function"]["name"] for item in TOOLS]
    assert names == [
        "bash", "read_file", "write_file", "edit_file", "glob",
        "read_image", "todo_write", "task",
    ]
    assert all(item["type"] == "function" for item in TOOLS)


def test_sub_tools_exclude_delegation_so_depth_stays_one():
    names = [item["function"]["name"] for item in SUB_TOOLS]
    assert "task" not in names
    assert names == [item["function"]["name"] for item in TOOLS if item["function"]["name"] != "task"]


def test_run_task_without_runner_is_refused():
    assert "requires an active parent agent loop" in run_task("do something", name="检查员")
    assert "requires an active parent agent loop" in dispatch_tool(
        "task", {"prompt": "do something", "name": "检查员"}, "/tmp"
    )
    assert "requires an active parent agent loop" in run_task("continue", session_id="child")
    assert "requires an active parent agent loop" in run_task(action="close", session_id="child")


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"prompt": "inspect", "name": "  影像检查员  "}, ("inspect", "run", None, "影像检查员", True, 0)),
        ({"action": "run", "prompt": "continue", "session_id": "child-123"}, ("continue", "run", "child-123", None, True, 0)),
        ({"action": "close", "session_id": "child-123"}, (None, "close", "child-123", None, True, 0)),
        ({"action": "status", "session_id": "child-123"}, (None, "status", "child-123", None, True, 0)),
        ({"action": "wait", "session_id": "child-123", "timeout_ms": 25}, (None, "wait", "child-123", None, True, 25)),
        ({"prompt": "x" * 8000, "name": "n" * 64}, ("x" * 8000, "run", None, "n" * 64, True, 0)),
    ],
)
def test_task_dispatches_valid_operations(arguments, expected):
    calls = []

    def runner(prompt, *, action, session_id, name, background, timeout_ms):
        calls.append((prompt, action, session_id, name, background, timeout_ms))
        return "task result"

    assert dispatch_tool("task", arguments, "/tmp", task_runner=runner) == "task result"
    assert calls == [expected]


@pytest.mark.parametrize("prompt", [None, "", "  ", 123, [], "x" * 8001])
def test_run_task_rejects_invalid_prompts_without_calling_runner(prompt):
    def runner(*args, **kwargs):
        pytest.fail("invalid prompt reached the runner")

    assert run_task(prompt, runner, name="检查员").startswith("Error:")


@pytest.mark.parametrize(
    "name", [None, "", "  ", 123, [], "n" * 65, "two\nlines", "name\n", "\tname", "a\x00b", "a\x7fb", "a\u200bb"]
)
def test_run_task_rejects_invalid_names_without_calling_runner(name):
    def runner(*args, **kwargs):
        pytest.fail("invalid name reached the runner")

    assert run_task("inspect", runner, name=name).startswith("Error:")


@pytest.mark.parametrize("session_id", ["", " ", ".", "..", "../child", "/child", "a/b", "child/", "a b", "a\nb", "a\x00b", 123, []])
def test_run_task_rejects_invalid_ids_without_creating_a_replacement(session_id):
    def runner(*args, **kwargs):
        pytest.fail("invalid session_id reached the runner")

    assert "Invalid session_id" in run_task("continue", runner, session_id=session_id)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"action": "unknown"},
        {"action": None},
        {"action": []},
        {"prompt": "inspect"},
        {"prompt": "continue", "session_id": "child", "name": "new name"},
        {"action": "close"},
        {"action": "close", "session_id": "child", "prompt": ""},
        {"action": "close", "session_id": "child", "name": ""},
        {"prompt": "inspect", "name": "inspector", "extra": True},
    ],
)
def test_task_rejects_invalid_argument_combinations(arguments):
    def runner(*args, **kwargs):
        pytest.fail("invalid arguments reached the runner")

    assert dispatch_tool("task", arguments, task_runner=runner).startswith("Error:")


def test_task_errors_and_large_results_remain_tool_observations():
    def failing_runner(*args, **kwargs):
        raise ValueError("failed to run child")

    assert run_task("inspect", failing_runner, name="inspector") == "Error: failed to run child"
    assert len(run_task("inspect", lambda *args, **kwargs: "x" * 60000, name="inspector")) == 50000


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


def test_read_image_loads_supported_image_and_enforces_workspace(tmp_path: Path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(JPEG)

    result = run_read_image("photo.jpg", workdir=tmp_path)

    assert isinstance(result, ImageAttachment)
    assert result.path == str(image.resolve())
    assert result.mime == "image/jpeg"
    assert run_read_image("missing.jpg", workdir=tmp_path).startswith("Error:")
    assert "Path escapes workspace" in run_read_image("../photo.jpg", workdir=tmp_path)

    raster = tmp_path / "scene.tif"
    raster.write_bytes(b"II*\x00opaque")
    assert "unsupported format" in run_read_image("scene.tif", workdir=tmp_path)


def test_bash_runs_in_workspace_and_truncates_output(tmp_path: Path):
    assert run_bash("pwd", tmp_path) == str(tmp_path.resolve())
    assert run_bash("sudo echo blocked", tmp_path) == "Error: Dangerous command blocked"
    assert run_bash("mkfs /dev/sda1", tmp_path) == "Error: Dangerous command blocked"
    assert run_bash("dd if=/dev/zero of=/tmp/img", tmp_path) == "Error: Dangerous command blocked"
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
