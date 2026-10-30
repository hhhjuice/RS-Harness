import io
import json
import subprocess
import sys

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from typer.testing import CliRunner

from rs_harness import application, cli
from rs_harness.agents.main import build_agent
from rs_harness.agents.skills import SkillMiddleware
from rs_harness.context import AgentContext
from rs_harness.progress import CliProgress
from rs_harness.sessions.checkpoints import open_checkpointer
from rs_harness.skills import SkillLoader
from tests.images import write_test_image
from tests.integration.test_subagents import call, delegate, install_worker, records, request, snapshot, tool_messages
from tests.models import ScriptedModel


BODY_MARKER = "FIXTURE-SKILL-BODY"


def put_skill(config, name, body):
    path = config.runtime.skills_dir / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: Description of {name}\n---\n{body}")
    return path


def load_call(name="fixture-skill", call_id="load"):
    return call("load_skill", {"name": name}, call_id)


def raw_skill(config):
    return (config.runtime.skills_dir / "fixture-skill/SKILL.md").read_text()


def assert_catalog_only(messages):
    systems = [message for message in messages if isinstance(message, SystemMessage)]
    assert len(systems) == 1
    text = str(systems[0].content)
    assert text.count("## Available Skills") == 1
    assert "fixture-skill" in text
    assert "Fixture skill for integration tests" in text
    assert BODY_MARKER not in text
    assert "FIXTURE-INSTALL-EXAMPLE" not in text


def test_sync_hook_preserves_system_blocks_and_registers_tool(skills_config, tmp_path):
    model = ScriptedModel([request(load_call()), AIMessage(content="done")])
    system = SystemMessage(content=[{"type": "text", "text": "BASE-ONE"}, {"type": "text", "text": "BASE-TWO"}])
    original = system.model_dump()
    graph = create_agent(model, system_prompt=system,
                         middleware=[SkillMiddleware(SkillLoader(skills_config.runtime.skills_dir))],
                         context_schema=AgentContext, checkpointer=InMemorySaver())
    run = {"configurable": {"thread_id": "sync"}}
    result = graph.invoke({"messages": [HumanMessage(content="load")]}, config=run,
                          context=AgentContext(str(tmp_path)))
    assert system.model_dump() == original
    for messages in model.inputs:
        assert_catalog_only(messages)
        assert "BASE-ONE" in str(messages[0].content) and "BASE-TWO" in str(messages[0].content)
    assert BODY_MARKER not in str(model.inputs[0])
    loaded = tool_messages(result["messages"])[0]
    assert loaded.tool_call_id == "load" and loaded.name == "load_skill" and loaded.status == "success"
    assert loaded.content.endswith(raw_skill(skills_config))
    schema = model.bound_tools[0].tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"name"}
    assert schema["required"] == ["name"]
    assert not any(isinstance(message, SystemMessage) for message in graph.get_state(run).values["messages"])


async def test_production_loading_composes_with_todo_images_and_repeated_calls(skills_config, tmp_path):
    write_test_image(tmp_path / "image.png")
    todos = [{"content": "Inspect skill", "status": "completed"}]
    model = ScriptedModel([
        request(load_call(), call("write_todos", {"todos": todos}, "todo"),
                call("read_image", {"path": "image.png"}, "image")),
        request(load_call(call_id="again")), AIMessage(content="done"),
    ])
    session = application.create_session(skills_config, workdir=str(tmp_path))
    result = await application.ask(skills_config, session.session_id, "inspect", model=model)
    assert result.status == "succeeded", result.error
    assert BODY_MARKER not in str(model.inputs[0])
    for messages in model.inputs:
        assert_catalog_only(messages)
        assert "You are a test assistant" in str(messages[0].content)
        assert "write_todos" in str(messages[0].content)
    loads = [message for message in tool_messages(model.inputs[-1]) if message.name == "load_skill"]
    assert [message.tool_call_id for message in loads] == ["load", "again"]
    assert all(message.content.endswith(raw_skill(skills_config)) for message in loads)
    assert sum(tool.name == "load_skill" for tool in model.bound_tools) == 1
    assert any(isinstance(message, HumanMessage) and isinstance(message.content, list)
               and any(block.get("type") == "image" for block in message.content) for message in model.inputs[-1])
    state = (await snapshot(skills_config, session)).values
    assert state["todos"] == todos
    assert sum(isinstance(message, HumanMessage) for message in state["messages"]) == 1
    assert not any(isinstance(message, SystemMessage) for message in state["messages"])


@pytest.mark.parametrize("args", [{"name": "missing"}, {"name": "../fixture-skill/SKILL.md"}, {"name": ""}, {}, {"name": 42}])
async def test_invalid_load_is_correctable_tool_error(skills_config, tmp_path, args):
    model = ScriptedModel([request(call("load_skill", args, "bad")), request(load_call()), AIMessage(content="done")])
    session = application.create_session(skills_config, workdir=str(tmp_path))
    result = await application.ask(skills_config, session.session_id, "run", model=model)
    assert result.status == "succeeded", result.error
    error, loaded = tool_messages(model.inputs[-1])
    assert error.status == "error" and error.tool_call_id == "bad"
    assert BODY_MARKER not in error.content
    assert loaded.status == "success" and loaded.content.endswith(raw_skill(skills_config))


async def test_parallel_loads_and_exact_utf8_output_budget(skills_config, tmp_path):
    put_skill(skills_config, "unicode", "内容" * 100)
    model = ScriptedModel([request(load_call("unicode", "one"), load_call("unicode", "two"), load_call()), AIMessage(content="done")])
    graph = build_agent(skills_config, workdir=str(tmp_path), model=model)
    result = await graph.ainvoke({"messages": [HumanMessage(content="run")]}, context=AgentContext(str(tmp_path)))
    outputs = tool_messages(result["messages"])
    assert [message.tool_call_id for message in outputs] == ["one", "two", "load"]
    assert outputs[0].content == outputs[1].content
    byte_count = len(outputs[0].content.encode("utf-8"))
    assert byte_count > len(outputs[0].content)
    for limit, status in ((byte_count, "success"), (byte_count - 1, "error")):
        bounded = ScriptedModel([request(load_call("unicode")), AIMessage(content="done")])
        graph = build_agent(skills_config, workdir=str(tmp_path), model=bounded)
        result = await graph.ainvoke({"messages": [HumanMessage(content="run")]},
                                    context=AgentContext(str(tmp_path), max_tool_output_bytes=limit))
        output = tool_messages(result["messages"])[0]
        assert output.status == status
        if status == "error":
            assert "max_tool_output_bytes" in output.content and "内容" not in output.content
        else:
            assert len(output.content.encode("utf-8")) == limit


async def test_resources_are_read_only_after_separate_tool_call(skills_config, tmp_path):
    path = put_skill(skills_config, "reference", "MANIFEST-DETAIL: consult reference.txt as needed")
    reference = path.parent / "reference.txt"
    reference.write_text("RESOURCE-DETAIL-7291")
    unrelated = ScriptedModel([AIMessage(content="2")])
    session = application.create_session(skills_config, workdir=str(tmp_path))
    result = await application.ask(skills_config, session.session_id, "1 + 1", model=unrelated)
    assert result.status == "succeeded"
    assert "MANIFEST-DETAIL" not in str(unrelated.inputs)
    model = ScriptedModel([request(load_call("reference")),
                           request(call("read_file", {"path": str(reference)}, "reference")), AIMessage(content="done")])
    result = await application.ask(skills_config, session.session_id, "consult", model=model)
    assert result.status == "succeeded", result.error
    assert "MANIFEST-DETAIL" not in str(model.inputs[0])
    assert "MANIFEST-DETAIL" in str(model.inputs[1]) and "RESOURCE-DETAIL-7291" not in str(model.inputs[1])
    assert "RESOURCE-DETAIL-7291" in str(model.inputs[2])
    assert f"Skill directory: {path.parent}" in tool_messages(model.inputs[1])[0].content


async def test_checkpoint_survives_source_removal_and_independent_process(skills_config, tmp_path):
    session = application.create_session(skills_config, workdir=str(tmp_path))
    first = ScriptedModel([request(load_call()), AIMessage(content="loaded")])
    result = await application.ask(skills_config, session.session_id, "load", model=first)
    assert result.status == "succeeded", result.error
    original = raw_skill(skills_config)
    (skills_config.runtime.skills_dir / "fixture-skill/SKILL.md").unlink()
    session.trajectory_path.unlink()
    script = '''
import asyncio, json, sys
from langchain_core.messages import AIMessage, ToolMessage
from rs_harness.application import ask
from rs_harness.config import Config
from tests.models import ScriptedModel
async def run():
    model = ScriptedModel([AIMessage(content="recalled")])
    result = await ask(Config.load(sys.argv[1]), sys.argv[2], "recall", model=model)
    tools = [m for m in model.inputs[0] if isinstance(m, ToolMessage)]
    print(json.dumps({"status": result.status, "tools": [{"id": m.tool_call_id, "content": m.content} for m in tools]}))
asyncio.run(run())
'''
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path / "config.toml"), session.session_id],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    restored = json.loads(process.stdout)
    assert restored["status"] == "succeeded"
    assert len(restored["tools"]) == 1 and restored["tools"][0]["id"] == "load"
    assert restored["tools"][0]["content"].endswith(original)
    other = application.create_session(skills_config, workdir=str(tmp_path))
    fresh = ScriptedModel([AIMessage(content="fresh")])
    await application.ask(skills_config, other.session_id, "hello", model=fresh)
    assert BODY_MARKER not in str(fresh.inputs)


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_external_library_load_and_resource_approval_resume(skills_config, tmp_path, decision):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    path = put_skill(skills_config, "reference", "MANIFEST-DETAIL: read reference.txt when needed")
    reference = path.parent / "reference.txt"
    reference.write_text("EXTERNAL-RESOURCE-8319")
    model = ScriptedModel([request(load_call("reference")),
                           request(call("read_file", {"path": str(reference)}, "reference")), AIMessage(content="done")])
    session = application.create_session(skills_config, workdir=str(workdir))
    pending = await application.ask(skills_config, session.session_id, "consult", model=model)
    assert pending.status == "interrupted", pending.error
    assert len(tool_messages(model.inputs[-1])) == 1
    assert "EXTERNAL-RESOURCE-8319" not in str(model.inputs)
    assert [action["name"] for action in pending.interrupts[0]["action_requests"]] == ["read_file"]
    result = await application.resume(skills_config, session.session_id, {
        pending.interrupts[0]["id"]: {"decisions": [{"type": decision}]},
    }, model=model)
    assert result.status == "succeeded", result.error
    outputs = tool_messages(model.inputs[-1])
    assert [message.tool_call_id for message in outputs] == ["load", "reference"]
    assert ("EXTERNAL-RESOURCE-8319" in outputs[-1].content) == (decision == "approve")
    assert outputs[-1].status == ("success" if decision == "approve" else "error")
    assert sum(row["event"] == "tool_start" and row["payload"]["name"] == "load_skill" for row in records(session)) == 1


@pytest.mark.parametrize("tools", [(), ("read_file",), ("load_skill",), ("load_skill", "write_todos")])
async def test_subagent_tool_and_catalog_whitelist(skills_config, tmp_path, monkeypatch, tools):
    config = skills_config.model_copy(update={"subagents": (skills_config.subagents[0].model_copy(update={"tools": tools}),)})
    worker = ScriptedModel([request(load_call()), AIMessage(content="checked")])
    parent = ScriptedModel([request(delegate()), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(config, session.session_id, "delegate", model=parent)
    assert result.status == "succeeded", result.error
    assert {tool.name for tool in worker.bound_tools} == set(tools)
    enabled = "load_skill" in tools
    assert ("## Available Skills" in str(worker.inputs[0][0])) == enabled
    output = tool_messages(worker.inputs[-1])[0]
    assert output.status == ("success" if enabled else "error")
    assert (BODY_MARKER in output.content) == enabled
    assert BODY_MARKER not in str(parent.inputs)


async def test_child_loads_external_manifest_but_cannot_read_external_reference(skills_config, tmp_path, monkeypatch):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    path = put_skill(skills_config, "reference", "MANIFEST-DETAIL")
    reference = path.parent / "reference.txt"
    reference.write_text("CHILD-MUST-NOT-READ")
    worker = ScriptedModel([request(load_call("reference")),
                            request(call("read_file", {"path": str(reference)}, "outside")), AIMessage(content="denied")])
    parent = ScriptedModel([request(delegate()), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(skills_config, workdir=str(workdir))
    result = await application.ask(skills_config, session.session_id, "delegate", model=parent)
    assert result.status == "succeeded", result.error
    outputs = tool_messages(worker.inputs[-1])
    assert outputs[0].status == "success" and "MANIFEST-DETAIL" in outputs[0].content
    assert outputs[1].status == "error" and "cannot request human approval" in outputs[1].content
    assert "CHILD-MUST-NOT-READ" not in str(worker.inputs)
    assert "MANIFEST-DETAIL" not in str(parent.inputs)


@pytest.mark.parametrize("persistence", ["per_invocation", "per_thread"])
async def test_skills_follow_parent_and_child_memory_boundaries(skills_config, tmp_path, monkeypatch, persistence):
    put_skill(skills_config, "main-only", "MAIN-PRIVATE-DETAIL")
    definitions = tuple(skills_config.subagents[0].model_copy(update={
        "id": name, "name": name, "tools": ("load_skill",), "persistence": persistence,
    }) for name in ("alpha", "beta"))
    config = skills_config.model_copy(update={"subagents": definitions})
    worker = ScriptedModel([request(load_call()), AIMessage(content="alpha checked"),
                            AIMessage(content="beta checked"), AIMessage(content="alpha recalled"),
                            AIMessage(content="new session")])
    parent = ScriptedModel([
        request(load_call("main-only")), request(delegate(agent_id="alpha", call_id="alpha")),
        request(delegate(agent_id="beta", call_id="beta")), AIMessage(content="done"),
        request(delegate(agent_id="alpha", call_id="followup")), AIMessage(content="continued"),
    ])
    install_worker(monkeypatch, worker)
    session = application.create_session(config, workdir=str(tmp_path))
    for text in ("start", "followup"):
        result = await application.ask(config, session.session_id, text, model=parent)
        assert result.status == "succeeded", result.error
    assert "MAIN-PRIVATE-DETAIL" not in str(worker.inputs)
    assert BODY_MARKER not in str(worker.inputs[0])
    assert BODY_MARKER in str(worker.inputs[1])
    assert BODY_MARKER not in str(worker.inputs[2])
    assert (BODY_MARKER in str(worker.inputs[3])) == (persistence == "per_thread")
    assert BODY_MARKER not in str(parent.inputs)
    async with open_checkpointer(config) as saver:
        checkpoints = [item async for item in saver.alist({"configurable": {"thread_id": session.session_id}})]
    assert any(item.config["configurable"]["checkpoint_ns"] and BODY_MARKER in str(item.checkpoint["channel_values"].get("messages", []))
               for item in checkpoints)
    other = application.create_session(config, workdir=str(tmp_path))
    fresh = ScriptedModel([request(delegate(agent_id="alpha")), AIMessage(content="done")])
    result = await application.ask(config, other.session_id, "start", model=fresh)
    assert result.status == "succeeded", result.error
    assert BODY_MARKER not in str(worker.inputs[-1])


async def test_one_snapshot_shared_by_main_and_child_refreshes_on_rebuild(skills_config, tmp_path, monkeypatch):
    original = raw_skill(skills_config)
    removed = put_skill(skills_config, "removed", "OLD-REMOVED-CONTENT")
    scans = []
    scan = SkillLoader._scan

    def count_scan(root):
        scans.append(root)
        return scan(root)

    monkeypatch.setattr(SkillLoader, "_scan", staticmethod(count_scan))
    parent = ScriptedModel([request(load_call(), load_call("removed", "removed"), load_call("added", "added")),
                            request(delegate()), AIMessage(content="done")])
    worker = ScriptedModel([request(load_call()), AIMessage(content="checked")])
    graph = build_agent(skills_config, workdir=str(tmp_path), model=parent, subagent_model=worker)
    assert len(scans) == 1
    path = skills_config.runtime.skills_dir / "fixture-skill/SKILL.md"
    path.write_text("---\nname: fixture-skill\ndescription: UPDATED-DESCRIPTION\n---\nNEW-CONTENT")
    removed.unlink()
    put_skill(skills_config, "added", "NEW-ADDED-CONTENT")
    await graph.ainvoke({"messages": [HumanMessage(content="run")]}, context=AgentContext(str(tmp_path)))
    assert len(scans) == 1
    assert tool_messages(parent.inputs[1])[0].content.endswith(original)
    assert tool_messages(worker.inputs[1])[0].content.endswith(original)
    assert [message.status for message in tool_messages(parent.inputs[1])] == ["success", "success", "error"]
    refreshed = ScriptedModel([request(load_call(), load_call("removed", "removed"), load_call("added", "added")), AIMessage(content="done")])
    new_graph = build_agent(skills_config, workdir=str(tmp_path), model=refreshed)
    await new_graph.ainvoke({"messages": [HumanMessage(content="run")]}, context=AgentContext(str(tmp_path)))
    assert len(scans) == 2
    assert "UPDATED-DESCRIPTION" in str(refreshed.inputs[0])
    outputs = tool_messages(refreshed.inputs[-1])
    assert [message.status for message in outputs] == ["success", "error", "success"]
    assert outputs[0].content.endswith("NEW-CONTENT") and outputs[2].content.endswith("NEW-ADDED-CONTENT")


async def test_progress_reports_main_loading_and_errors_without_child_or_body(skills_config, tmp_path, monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr("rs_harness.progress.sys.stderr", stream)
    worker = ScriptedModel([request(load_call()), AIMessage(content="checked")])
    parent = ScriptedModel([request(load_call()), request(load_call("missing", "bad")),
                            request(delegate()), AIMessage(content="done")])
    install_worker(monkeypatch, worker)
    session = application.create_session(skills_config, workdir=str(tmp_path))
    result = await application.ask(skills_config, session.session_id, "load", model=parent, callbacks=[CliProgress()])
    assert result.status == "succeeded", result.error
    output = stream.getvalue()
    assert output.count("[Tool] Running load_skill") == 2
    assert output.count("[Tool] Completed load_skill") == 1
    assert output.count("[Tool] Failed load_skill") == 1
    assert '"name":"fixture-skill"' in output and "Unknown skill" in output
    assert BODY_MARKER not in output and "Loaded skill:" not in output
    events = records(session)
    assert sum(row["event"] == "tool_start" and row["payload"]["name"] == "load_skill" for row in events) == 3


def test_cli_json_contains_one_result_and_no_skill_body(skills_config, tmp_path, monkeypatch):
    model = ScriptedModel([request(load_call()), AIMessage(content="done")])

    async def with_model(config, session_id, text, **kwargs):
        return await application.ask(config, session_id, text, model=model, **kwargs)

    monkeypatch.setattr(cli, "ask", with_model)
    result = CliRunner().invoke(cli.app, ["ask", "load", "--config", str(tmp_path / "config.toml"),
                                          "--workdir", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "succeeded" and payload["answer"] == "done"
    assert result.stderr == "" and BODY_MARKER not in result.stdout


async def test_skill_loading_does_not_reset_model_budget(skills_config, tmp_path):
    config = skills_config.model_copy(update={"runtime": skills_config.runtime.model_copy(update={"max_model_calls": 1})})
    session = application.create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([request(load_call()), AIMessage(content="continued")])
    failed = await application.ask(config, session.session_id, "load", model=model)
    assert failed.status == "failed" and failed.error["type"] == "ModelCallLimitExceededError"
    assert model.call_count == 1
    state = (await snapshot(config, session)).values
    assert tool_messages(state["messages"])[0].content.endswith(raw_skill(config))
    result = await application.ask(config, session.session_id, "continue", model=model)
    assert result.status == "succeeded" and model.call_count == 2
    assert BODY_MARKER in str(model.inputs[-1])
