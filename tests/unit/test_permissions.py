from types import SimpleNamespace

import pytest

from rs_harness.agents.middleware import ToolDenyMiddleware, classify_tool_call, needs_approval
from rs_harness.context import AgentContext


@pytest.mark.parametrize(("command", "expected"), [
    ("rm -rf /", "deny"), ("sudo rm file.txt", "deny"), ("shutdown", "deny"),
    ("reboot", "deny"), ("mkfs.ext4 disk", "deny"), ("dd if=image", "deny"),
    ("cat image > /dev/sda", "deny"), ("rm -rf /tmp/demo", "deny"),
    ("echo 'sudo'", "deny"), ("rm cleanup.txt", "ask"),
    ("echo x > /etc/test", "ask"), ("chmod 777 file", "ask"),
    ("pwd", "allow"), ("cat note.txt", "allow"), ("pytest", "allow"),
    ("printf x >> note.txt", "allow"), ("SUDO", "allow"), ("rm\tfile", "allow"),
])
def test_literal_bash_policy(tmp_path, command, expected):
    decision, reason = classify_tool_call({"name": "bash", "args": {"command": command}}, str(tmp_path))
    assert decision == expected
    assert bool(reason) == (expected != "allow")


@pytest.mark.parametrize("name", ["read_file", "write_file", "edit_file", "read_image"])
@pytest.mark.parametrize("kind", ["relative", "absolute", "symlink", "symlink_parent", "inside"])
def test_file_policy_uses_resolved_session_workdir(tmp_path, name, kind):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    target = tmp_path / "outside.txt"
    if kind == "relative":
        path = "../outside.txt"
    elif kind == "absolute":
        path = str(target)
    elif kind == "symlink":
        (workdir / "link").symlink_to(target)
        path = "link"
    elif kind == "symlink_parent":
        (workdir / "link").symlink_to(tmp_path, target_is_directory=True)
        path = "link/outside.txt"
    else:
        path = "nested/new.txt"
    call = {"name": name, "args": {"path": path, "workdir": str(tmp_path), "approved": True}}
    decision, reason = classify_tool_call(call, str(workdir))
    assert decision == ("allow" if kind == "inside" else "ask")
    if kind != "inside":
        assert str(target.resolve()) in reason
    request = SimpleNamespace(tool=None, tool_call=call, runtime=SimpleNamespace(context=AgentContext(str(workdir))))
    assert needs_approval(request) == (decision == "ask")
    assert not target.exists()


@pytest.mark.parametrize("arguments", [{}, {"path": None}, {"path": 3}, {"path": []}, {"path": ""}, {"path": "\x00"}, {"command": None}, {"command": 4}])
@pytest.mark.parametrize("name", ["read_file", "write_file", "edit_file", "read_image", "bash", "glob"])
def test_invalid_arguments_are_left_to_tool_validation(tmp_path, name, arguments):
    assert classify_tool_call({"name": name, "args": arguments}, str(tmp_path)) == ("allow", "")


@pytest.mark.parametrize("command", ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"])
async def test_hard_deny_never_calls_handler(tmp_path, command):
    async def handler(request):
        pytest.fail("Forbidden tool handler was called")

    request = SimpleNamespace(
        tool_call={"name": "bash", "args": {"command": command}, "id": "blocked"},
        runtime=SimpleNamespace(context=AgentContext(str(tmp_path))),
    )
    result = await ToolDenyMiddleware().awrap_tool_call(request, handler)
    assert result.status == "error"
    assert result.tool_call_id == "blocked"
    assert result.name == "bash"
    assert command in result.content
