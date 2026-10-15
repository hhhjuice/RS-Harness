import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from rs_harness.agents.main import build_agent_components
from rs_harness.application import ask, create_session, get_context_usage, resume
from rs_harness.config import AgentConfig, MAIN_AGENT_TOOLS
from rs_harness.skills import SkillLoader
from tests.integration.test_mcp import install_servers, response, successful_calls
from tests.integration.test_permissions import call, decide, snapshot
from tests.integration.test_skills import put_skill
from tests.integration.test_subagents import install_worker
from tests.models import ScriptedModel
from tests.unit.test_mcp import settings


@pytest.mark.parametrize("selected", [
    (), *((name,) for name in MAIN_AGENT_TOOLS),
    ("read_image", "glob", "read_file"), ("write_todos", "load_skill"), MAIN_AGENT_TOOLS,
])
async def test_selected_tools_match_model_and_context(config, tmp_path, selected):
    config = config.model_copy(update={"agent": AgentConfig(tools=selected)})
    put_skill(config, "visible-skill", "PRIVATE-SKILL-BODY")
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="Hello")])

    result = await ask(config, session.session_id, "Hello", model=model)

    assert result.status == "succeeded" and result.answer == "Hello", result.error
    assert {tool.name for tool in model.bound_tools} == set(selected)
    components = build_agent_components(config, workdir=str(tmp_path))
    request = components.context_request((await snapshot(config, session)).values)
    assert [convert_to_openai_tool(tool) for tool in request.tools] == [
        convert_to_openai_tool(tool) for tool in model.tool_history[0]
    ]
    assert request.system_message.content == model.inputs[0][0].content
    system = str(request.system_message.content)
    assert ("Available Skills" in system) == ("load_skill" in selected)
    assert ("visible-skill" in system) == ("load_skill" in selected)
    assert ("write_todos" in system) == ("write_todos" in selected)
    assert "PRIVATE-SKILL-BODY" not in system
    middleware = {item.name: item for item in components.middleware}
    assert ("SubagentBatchMiddleware" in middleware) == ("delegate_task" in selected)
    assert ("TodoListMiddleware" in middleware) == ("write_todos" in selected)
    assert ("SkillMiddleware" in middleware) == ("load_skill" in selected)
    assert middleware["MCPMiddleware"].registry.local_names == set(MAIN_AGENT_TOOLS) | {"connect_mcp"}
    usage = await get_context_usage(config, session.session_id)
    counts = {item.key: item.tokens for item in usage.categories}
    assert (counts["tool_schemas"] > 0) == bool(selected)
    assert (counts["skills"] > 0) == ("load_skill" in selected)


async def test_disabled_tools_cannot_execute_or_inject_prompts(config, tmp_path, monkeypatch):
    config = config.model_copy(update={"agent": AgentConfig(tools=("read_file",))})
    (tmp_path / "input.txt").write_text("READ-EVIDENCE")
    put_skill(config, "hidden", "PRIVATE-SKILL-BODY")

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled capabilities must not execute or build subagent graphs")

    monkeypatch.setattr("rs_harness.agents.main.build_delegate_tool", forbidden)
    monkeypatch.setattr(SkillLoader, "load", forbidden)
    model = ScriptedModel([
        response(
            call("read_file", {"path": "input.txt"}, "allowed"),
            call("write_file", {"path": "blocked.txt", "content": "no"}, "write"),
            call("edit_file", {"path": "input.txt", "old_text": "READ-EVIDENCE", "new_text": "no"}, "edit"),
            call("bash", {"command": "printf no > shell.txt"}, "shell"),
            call("write_todos", {"todos": [{"content": "no", "status": "completed"}]}, "todo"),
            call("load_skill", {"name": "hidden"}, "skill"),
            call("delegate_task", {"agent_id": "worker", "query": "Write a file"}, "delegate"),
        ),
        AIMessage(content="Only the read succeeded"),
    ])
    session = create_session(config, workdir=str(tmp_path))

    result = await ask(config, session.session_id, "Read only", model=model)

    assert result.status == "succeeded", result.error
    messages = {message.tool_call_id: message for message in model.inputs[-1] if isinstance(message, ToolMessage)}
    assert messages["allowed"].status == "success" and "READ-EVIDENCE" in messages["allowed"].content
    assert all(messages[name].status == "error" for name in ("write", "edit", "shell", "todo", "skill", "delegate"))
    assert not (tmp_path / "blocked.txt").exists() and not (tmp_path / "shell.txt").exists()
    assert (tmp_path / "input.txt").read_text() == "READ-EVIDENCE"
    assert not (await snapshot(config, session)).values.get("todos")
    assert "PRIVATE-SKILL-BODY" not in str(model.inputs)


@pytest.mark.parametrize("selected", [MAIN_AGENT_TOOLS, ("delegate_task",)])
async def test_missing_subagents_skip_delegation(config, tmp_path, monkeypatch, selected):
    config = config.model_copy(update={"agent": AgentConfig(tools=selected), "subagents": ()})

    def forbidden(*args, **kwargs):
        pytest.fail("No delegate tool should be built without subagent definitions")

    monkeypatch.setattr("rs_harness.agents.main.build_delegate_tool", forbidden)
    monkeypatch.setattr("rs_harness.agents.main.make_delegate_tool", forbidden)
    model = ScriptedModel([AIMessage(content="Hello")])
    session = create_session(config, workdir=str(tmp_path))

    result = await ask(config, session.session_id, "Hello", model=model)

    assert result.status == "succeeded", result.error
    assert {tool.name for tool in model.bound_tools} == set(selected) - {"delegate_task"}
    components = build_agent_components(config, workdir=str(tmp_path))
    assert "SubagentBatchMiddleware" not in {item.name for item in components.middleware}
    assert {tool.name for tool in components.context_request({}).tools} == set(selected) - {"delegate_task"}


async def test_delegation_keeps_child_tool_permissions_independent(config, tmp_path, monkeypatch):
    child_tools = ("write_file", "write_todos", "load_skill")
    config = config.model_copy(update={
        "agent": AgentConfig(tools=("delegate_task",)),
        "subagents": (config.subagents[0].model_copy(update={"tools": child_tools}),),
    })
    put_skill(config, "child-skill", "CHILD-SKILL-EVIDENCE")
    child = ScriptedModel([
        response(
            call("write_file", {"path": "child.txt", "content": "child-evidence"}, "child-write"),
            call("write_todos", {"todos": [{"content": "Write evidence", "status": "completed"}]}, "child-todo"),
            call("load_skill", {"name": "child-skill"}, "child-skill"),
        ),
        AIMessage(content="Child completed"),
    ])
    install_worker(monkeypatch, child)
    parent = ScriptedModel([
        response(call("delegate_task", {"agent_id": "worker", "query": "Write evidence and load your skill"}, "delegate")),
        response(call("write_file", {"path": "parent.txt", "content": "no"}, "parent-write")),
        AIMessage(content="Done"),
    ])
    session = create_session(config, workdir=str(tmp_path))

    result = await ask(config, session.session_id, "Delegate", model=parent)

    assert result.status == "succeeded", result.error
    assert (tmp_path / "child.txt").read_text() == "child-evidence"
    assert not (tmp_path / "parent.txt").exists()
    assert {tool.name for tool in parent.bound_tools} == {"delegate_task"}
    assert {tool.name for tool in child.bound_tools} == set(child_tools)
    assert "Available Skills" not in str(parent.inputs[0][0].content)
    assert "Available Skills" in str(child.inputs[0][0].content)
    child_results = [message for message in child.inputs[-1] if isinstance(message, ToolMessage)]
    assert len(child_results) == 3 and all(message.status == "success" for message in child_results)
    assert "CHILD-SKILL-EVIDENCE" in str(child_results)


@pytest.mark.parametrize("approval", [False, True])
async def test_empty_builtin_list_keeps_mcp_tools_and_approval(config, tmp_path, monkeypatch, approval):
    install_servers(monkeypatch, tmp_path, ("docs",))
    config = config.model_copy(update={
        "agent": AgentConfig(tools=()),
        "mcp": settings(docs={"tools": {"search": "confirm"} if approval else {}}),
    })
    model = ScriptedModel([
        response(call("connect_mcp", {"name": "docs"}, "connect")),
        response(call("mcp__docs__search", {"query": "evidence"}, "search")),
        AIMessage(content="MCP completed"),
    ])
    session = create_session(config, workdir=str(tmp_path))

    result = await ask(config, session.session_id, "Search MCP", model=model)

    if approval:
        assert result.status == "interrupted", result.error
        assert successful_calls(tmp_path / "docs.jsonl") == []
        assert [item["name"] for item in result.interrupts[0]["action_requests"]] == ["mcp__docs__search"]
        result = await resume(config, session.session_id, decide(result, {"type": "approve"}), model=model)
    assert result.status == "succeeded", result.error
    assert successful_calls(tmp_path / "docs.jsonl") == ["search"]
    assert {tool.name for tool in model.tool_history[0]} == {"connect_mcp"}
    assert {tool.name for tool in model.bound_tools} == {"connect_mcp", "mcp__docs__search", "mcp__docs__get_version"}
    request = build_agent_components(config, workdir=str(tmp_path)).context_request((await snapshot(config, session)).values)
    actual = [convert_to_openai_tool(tool) for tool in request.tools]
    expected = [convert_to_openai_tool(tool) for tool in model.bound_tools]
    # Saved MCP schemas are sorted; discovery order need not be alphabetical.
    assert {item["function"]["name"]: item for item in actual} == {
        item["function"]["name"]: item for item in expected
    }
    assert "Available Skills" not in str(request.system_message.content)
    assert "write_todos" not in str(request.system_message.content)
