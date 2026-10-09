import pytest
from langchain_core.messages import AIMessage

from rs_harness import application
from rs_harness.progress import CliProgress
from tests.images import write_test_image
from tests.integration.test_subagents import call, delegate, install_worker, records, request, tool_messages
from tests.models import ScriptedModel


@pytest.mark.parametrize("name", ["read_file", "read_image", "write_file", "edit_file"])
@pytest.mark.parametrize("path_kind", ["absolute", "parent", "symlink", "sibling_prefix"])
async def test_child_external_files_are_denied_without_approval(config, tmp_path, monkeypatch, name, path_kind):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    outside = tmp_path / "workspace-other"
    outside.mkdir()
    target = outside / "private.png"
    if name == "read_image":
        write_test_image(target)
    else:
        target.write_text("outside-secret-7319")
    original = target.read_bytes()
    if path_kind == "parent":
        path = "../workspace-other/private.png"
    elif path_kind == "symlink":
        (workdir / "link").symlink_to(outside, target_is_directory=True)
        path = "link/private.png"
    elif path_kind == "sibling_prefix":
        path = str(target)
    else:
        target = tmp_path / "private.png"
        target.write_bytes(original)
        path = str(target)
    args = {"path": path}
    if name == "write_file":
        args["content"] = "changed"
    elif name == "edit_file":
        args.update(old_text="outside-secret-7319", new_text="changed")
    worker = ScriptedModel([request(call(name, args, "outside")), AIMessage(content="The action was denied")])
    parent = ScriptedModel([request(delegate()), AIMessage(content="reported")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(workdir))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded", result.error
    assert not result.interrupts
    assert not (await application.get_pending_approvals(config, session.session_id, model=parent)).interrupts
    assert target.read_bytes() == original
    error = tool_messages(worker.inputs[-1])[0]
    assert error.status == "error" and error.tool_call_id == "outside"
    assert "denied" in error.content.lower() and "outside-secret-7319" not in error.content
    assert not any(row["event"] == "tool_start" and row["payload"]["name"] == name for row in records(session))


async def test_child_progress_is_hidden_but_tool_and_parent_remain_visible(config, tmp_path, monkeypatch, capsys):
    worker = ScriptedModel([
        request(call("write_todos", {"todos": [{"content": "private-plan", "status": "completed"}]}, "todo"),
                call("bash", {"command": "sudo touch forbidden"}, "deny"),
                call("read_file", {"path": "private-missing-file"}, "missing")),
        AIMessage(content="private-child-result"),
    ])
    parent = ScriptedModel([
        request(delegate("private-query"), call("write_file", {"path": "parent-artifact", "content": "kept"}, "main")),
        AIMessage(content="reported"),
    ])
    for message in worker.responses:
        message.additional_kwargs["reasoning_content"] = "private-child-thinking"
    for index, message in enumerate(parent.responses):
        message.additional_kwargs["reasoning_content"] = f"parent thought {index}"
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    progress = CliProgress()
    result = await application.ask(config, session.session_id, "run", model=parent, callbacks=[progress])
    assert result.status == "succeeded", result.error
    output = capsys.readouterr().err
    assert "[SubAgent" not in output
    assert not any(text in output for text in ("private-", "[Todo]", "[Middleware]", "read_file", "bash", "forbidden"))
    assert output.count("[Tool] Running delegate_task\n") == 1
    assert output.count("[Tool] Completed delegate_task (") == 1
    assert output.index("[Tool] Running delegate_task\n") < output.index("[Tool] Completed delegate_task (")
    assert output.count("[Tool] Completed write_file (") == 1
    assert [line for line in output.splitlines() if line.startswith("[Thinking]")] == ["[Thinking] parent thought 0", "[Thinking] parent thought 1"]
    assert "[Model]" not in output and not progress._model_runs
    assert not progress._hidden_runs and not progress._tools and not progress._middleware
    assert any(row["event"] == "tool_start" and row["payload"]["name"] == "write_todos" for row in records(session))
    assert tool_messages(parent.inputs[-1])[0].content == "private-child-result"


@pytest.mark.parametrize("command, allowed", [
    ("printf changed > target", True),
    ("rm target", False),
    ("chmod 777 target", False),
    ("printf '> /etc/rsh-policy-test'", False),
    ("sudo touch forbidden", False),
])
async def test_child_bash_maps_allow_ask_deny_to_execute_deny_deny(config, tmp_path, monkeypatch, command, allowed):
    target = tmp_path / "target"
    target.write_text("original")
    original_mode = target.stat().st_mode
    worker = ScriptedModel([request(call("bash", {"command": command}, "shell")), AIMessage(content="reported")])
    parent = ScriptedModel([request(delegate()), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded" and not result.interrupts, result.error
    assert tool_messages(worker.inputs[-1])[0].status == ("success" if allowed else "error")
    assert target.read_text() == ("changed" if allowed else "original")
    assert target.stat().st_mode == original_mode and not (tmp_path / "forbidden").exists()
    executed = [row for row in records(session) if row["event"] == "tool_start" and row["payload"]["name"] == "bash"]
    assert len(executed) == int(allowed)


async def test_child_can_still_use_in_workspace_file_tools(config, tmp_path, monkeypatch):
    write_test_image(tmp_path / "image.png")
    worker = ScriptedModel([
        request(call("write_file", {"path": "result.txt", "content": "before"}, "write")),
        request(call("edit_file", {"path": "result.txt", "old_text": "before", "new_text": "after"}, "edit")),
        request(call("read_file", {"path": "result.txt"}, "read"),
                call("read_image", {"path": "image.png"}, "image"), call("glob", {"pattern": "*.txt"}, "glob")),
        AIMessage(content="implemented"),
    ])
    parent = ScriptedModel([request(delegate()), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "run", model=parent)
    assert result.status == "succeeded" and not result.interrupts, result.error
    assert (tmp_path / "result.txt").read_text() == "after"
    assert all(message.status == "success" for message in tool_messages(worker.inputs[-1]))


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_parent_may_decide_to_act_after_child_denial_using_its_own_approval(config, tmp_path, monkeypatch, decision):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    external = tmp_path / "external"
    external.write_text("original")
    worker = ScriptedModel([
        request(call("write_file", {"path": "../external", "content": "child"}, "child")), AIMessage(content="denied"),
    ])
    parent = ScriptedModel([
        request(delegate()),
        request(call("write_file", {"path": "../external", "content": "parent"}, "parent")),
        AIMessage(content="reported"),
    ])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(workdir))
    pending = await application.ask(config, session.session_id, "run", model=parent)
    assert pending.status == "interrupted", pending.error
    assert worker.call_count == parent.call_count == 2
    assert tool_messages(worker.inputs[-1])[0].status == "error"
    assert tool_messages(parent.inputs[-1])[0].content == "denied"
    assert pending.interrupts[0]["action_requests"][0]["args"]["content"] == "parent"
    assert external.read_text() == "original"
    final = await application.resume(config, session.session_id, {
        pending.interrupts[0]["id"]: {"decisions": [{"type": decision}]},
    }, model=parent)
    assert final.status == "succeeded", final.error
    assert final.interaction_id == pending.interaction_id and worker.call_count == 2
    assert external.read_text() == ("parent" if decision == "approve" else "original")
