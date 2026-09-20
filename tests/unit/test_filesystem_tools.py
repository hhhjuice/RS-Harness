from types import SimpleNamespace

import pytest
from langchain_core.tools import ToolException

from rs_harness.context import AgentContext
from rs_harness.tools.filesystem import edit_file, glob, read_file, write_file


@pytest.fixture
def runtime(tmp_path):
    return SimpleNamespace(context=AgentContext(str(tmp_path)))


def test_write_read_edit_preserve_content(tmp_path, runtime):
    content = "第一行\r\nsecond\r\n"
    written = write_file.func("sample.txt", content, runtime)
    assert written["bytes_written"] == len(content.encode())
    assert (tmp_path / "sample.txt").read_bytes() == content.encode()
    assert read_file.func("sample.txt", runtime, offset=2, limit=1)["content"] == "second\r\n"
    assert edit_file.func("sample.txt", "second", "", runtime)["replacements"] == 1
    assert (tmp_path / "sample.txt").read_bytes() == "第一行\r\n\r\n".encode()
    write_file.func("sample.txt", "", runtime)
    assert read_file.func("sample.txt", runtime)["content"] == ""


@pytest.mark.parametrize("old_text", ["missing", "same"])
def test_edit_failure_does_not_write(tmp_path, runtime, old_text):
    (tmp_path / "sample.txt").write_text("same same")
    with pytest.raises(ToolException, match="exactly once"):
        edit_file.func("sample.txt", old_text, "changed", runtime)
    assert (tmp_path / "sample.txt").read_text() == "same same"


def test_read_limits_and_unicode(tmp_path, runtime):
    (tmp_path / "sample.txt").write_text("你好世界\nnext\n")
    runtime.context = AgentContext(str(tmp_path), max_tool_output_bytes=7)
    result = read_file.func("sample.txt", runtime)
    assert result["content"] == "你好"
    assert result["truncated"] is True
    assert result["start_line"] == result["end_line"] == 1
    assert read_file.func("sample.txt", runtime, offset=20)["end_line"] is None
    runtime.context = AgentContext(str(tmp_path))
    result = read_file.func("sample.txt", runtime, limit=1)
    assert result["content"] == "你好世界\n"
    assert result["truncated"] is True


@pytest.mark.parametrize("kind", ["parent", "absolute", "symlink", "symlink_parent"])
def test_paths_cannot_escape_workdir(tmp_path, runtime, kind):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "file.txt").write_text("untouched")
    if kind == "parent":
        path = f"../{outside.name}/file.txt"
    elif kind == "absolute":
        path = str(outside / "file.txt")
    elif kind == "symlink":
        (tmp_path / "link").symlink_to(outside / "file.txt")
        path = "link"
    else:
        (tmp_path / "link").symlink_to(outside, target_is_directory=True)
        path = "link/new.txt"
    for tool, arguments in [
        (read_file, {"path": path}),
        (write_file, {"path": path, "content": "changed"}),
        (edit_file, {"path": path, "old_text": "untouched", "new_text": "changed"}),
    ]:
        with pytest.raises(ToolException, match="outside"):
            tool.func(**arguments, runtime=runtime)
    assert (outside / "file.txt").read_text() == "untouched"
    assert not (outside / "new.txt").exists()


def test_invalid_files_return_errors(tmp_path, runtime):
    (tmp_path / "binary").write_bytes(b"\x00\xff")
    for path in ["missing", ".", "binary"]:
        with pytest.raises(ToolException):
            read_file.func(path, runtime)
    with pytest.raises(ToolException):
        write_file.func("missing/file", "content", runtime)


def test_glob_sorting_hidden_files_and_boundary(tmp_path, runtime):
    (tmp_path / "nested").mkdir()
    for path in ["z.txt", ".hidden.txt", "nested/a.txt"]:
        (tmp_path / path).write_text("")
    (tmp_path / "outside.txt").symlink_to(tmp_path.parent / "elsewhere")
    assert glob.func("**/*.txt", runtime) == {
        "paths": [".hidden.txt", "nested/a.txt", "z.txt"], "truncated": False,
    }
    assert glob.func("*.absent", runtime)["paths"] == []
    for pattern in ["../*", str(tmp_path / "*")]:
        with pytest.raises(ToolException):
            glob.func(pattern, runtime)
    runtime.context = AgentContext(str(tmp_path), max_tool_output_bytes=1)
    assert glob.func("**/*.txt", runtime) == {"paths": [], "truncated": True}


def test_glob_count_limit(tmp_path, runtime):
    for index in range(201):
        (tmp_path / f"{index:03}.txt").touch()
    result = glob.func("*.txt", runtime)
    assert len(result["paths"]) == 200
    assert result["truncated"] is True


def test_read_line_numbers_at_exact_byte_limit(tmp_path, runtime):
    (tmp_path / "sample.txt").write_text("a\nb\n")
    runtime.context = AgentContext(str(tmp_path), max_tool_output_bytes=2)
    result = read_file.func("sample.txt", runtime)
    assert result["content"] == "a\n"
    assert result["end_line"] == 1
    assert result["truncated"] is True


def test_offset_handles_crlf_split_at_buffer_boundary(tmp_path, runtime):
    (tmp_path / "sample.txt").write_bytes(b"a" * 8191 + b"\r\nsecond\r\n")
    assert read_file.func("sample.txt", runtime, offset=2)["content"] == "second\r\n"
