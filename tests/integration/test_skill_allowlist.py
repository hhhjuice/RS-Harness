import pytest
from langchain_core.messages import AIMessage, HumanMessage
from typer.testing import CliRunner

from rs_harness import cli
from rs_harness.agents.main import build_agent, build_agent_components
from rs_harness.application import ask, create_session, get_context_usage
from rs_harness.config import AgentConfig, ConfigError
from rs_harness.context import AgentContext
from rs_harness.context_usage import estimate_categories
from rs_harness.skills import SkillLoader
from tests.integration.test_mcp import response
from tests.integration.test_permissions import call
from tests.integration.test_skills import put_skill
from tests.integration.test_subagents import install_worker, tool_messages
from tests.models import ScriptedModel


def load(name):
    return call("load_skill", {"name": name}, name)


@pytest.mark.parametrize("selected", [None, (), ("alpha",), ("beta", "alpha")])
async def test_catalog_loading_and_context_obey_main_selection(config, tmp_path, selected):
    config = config.model_copy(update={"agent": AgentConfig(tools=("load_skill",), skills=selected)})
    bodies = {name: f"PRIVATE-{name}-BODY" for name in ("alpha", "beta", "hidden")}
    paths = {name: put_skill(config, name, body) for name, body in bodies.items()}
    allowed = set(bodies) if selected is None else set(selected)
    components = build_agent_components(config, workdir=str(tmp_path))
    request = components.context_request({})
    session = create_session(config, workdir=str(tmp_path))
    usage = await get_context_usage(config, session.session_id)
    expected = estimate_categories(request.system_message, [], request.tools, skills_prompt=components.skills_prompt)
    assert next(item.tokens for item in usage.categories if item.key == "skills") == expected["skills"]
    model = ScriptedModel([
        response(*(load(name) for name in (*bodies, "unknown"))),
        AIMessage(content="Finished"),
    ])

    result = await ask(config, session.session_id, "Load the skills", model=model)

    assert result.status == "succeeded", result.error
    assert {tool.name for tool in model.bound_tools} == {"load_skill"}
    assert request.system_message.content == model.inputs[0][0].content
    results = {message.tool_call_id: message for message in tool_messages(model.inputs[-1])}
    for name, body in bodies.items():
        assert (f"- {name}: Description of {name}" in components.skills_prompt) == (name in allowed)
        if name in allowed:
            assert results[name].status == "success" and results[name].content.endswith(body)
        else:
            assert results[name].status == "error"
            assert body not in str(model.inputs)
            assert str(paths[name]) not in str(model.inputs)
            assert f"Description of {name}" not in str(model.inputs)
            assert name not in results["unknown"].content
    assert results["unknown"].status == "error"
    if not allowed:
        assert "## Available Skills\n\nnone\n" in components.skills_prompt
        assert all(message.status == "error" for message in results.values())


async def test_skill_selection_does_not_enable_load_skill_tool(config, tmp_path):
    put_skill(config, "alpha", "PRIVATE-ALPHA")
    config = config.model_copy(update={"agent": AgentConfig(tools=(), skills=("alpha",))})
    model = ScriptedModel([AIMessage(content="Hello")])
    session = create_session(config, workdir=str(tmp_path))

    result = await ask(config, session.session_id, "Hello", model=model)

    assert result.status == "succeeded", result.error
    assert model.bound_tools == []
    components = build_agent_components(config, workdir=str(tmp_path))
    assert components.skills_prompt == ""
    assert "Available Skills" not in str(model.inputs[0][0].content)
    usage = await get_context_usage(config, session.session_id)
    assert next(item.tokens for item in usage.categories if item.key == "skills") == 0


@pytest.mark.parametrize("owner", ["main", "worker"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("source", ["missing", "invalid"])
async def test_unknown_skills_fail_before_graph_construction_and_context(config, tmp_path, monkeypatch, owner, enabled, source):
    if source == "invalid":
        path = put_skill(config, "unavailable", "PRIVATE-INVALID-BODY")
        path.write_text("---\nname: [\n---\nPRIVATE-INVALID-BODY")
    agent = AgentConfig(skills=("unavailable",) if owner == "main" else None)
    child = config.subagents[0].model_copy(update={"skills": ("unavailable",) if owner == "worker" else None})
    if not enabled:
        agent = agent.model_copy(update={"tools": ()})
        child = child.model_copy(update={"tools": ()})
    config = config.model_copy(update={"agent": agent, "subagents": (child,)})

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid skill selections must be rejected before constructing any Agent graph")

    monkeypatch.setattr("langchain.agents.create_agent", forbidden)
    label = "Main agent" if owner == "main" else "Subagent worker"
    for model in (None, ScriptedModel([AIMessage(content="unused")])):
        with pytest.raises(ConfigError, match=f"{label}.*unavailable") as error:
            build_agent_components(config, workdir=str(tmp_path), model=model)
        assert "PRIVATE-INVALID-BODY" not in str(error.value)
    session = create_session(config, workdir=str(tmp_path))
    with pytest.raises(ConfigError, match=f"{label}.*unavailable"):
        await get_context_usage(config, session.session_id)


async def test_main_and_multiple_children_have_independent_skill_views(config, tmp_path, monkeypatch):
    for name in ("alpha", "beta", "gamma"):
        put_skill(config, name, f"PRIVATE-{name}-BODY")
    children = tuple(config.subagents[0].model_copy(update={
        "id": agent_id, "skills": (name,), "tools": ("load_skill",),
    }) for agent_id, name in (("worker", "beta"), ("other", "gamma")))
    config = config.model_copy(update={
        "agent": AgentConfig(tools=("load_skill", "delegate_task"), skills=("alpha",)),
        "subagents": children,
    })
    worker = ScriptedModel([
        response(load("beta"), load("alpha")), AIMessage(content="First child finished"),
        response(load("gamma"), load("beta")), AIMessage(content="Second child finished"),
    ])
    install_worker(monkeypatch, worker)
    parent = ScriptedModel([
        response(load("alpha"), load("beta")),
        response(call("delegate_task", {"agent_id": "worker", "query": "Load your skill"}, "worker")),
        response(call("delegate_task", {"agent_id": "other", "query": "Load your skill"}, "other")),
        AIMessage(content="Done"),
    ])
    session = create_session(config, workdir=str(tmp_path))

    result = await ask(config, session.session_id, "Check permissions", model=parent)

    assert result.status == "succeeded", result.error
    for model, before, after, permitted in ((parent, 0, 1, "alpha"), (worker, 0, 1, "beta"), (worker, 2, 3, "gamma")):
        system = str(model.inputs[before][0].content)
        assert f"Description of {permitted}" in system
        assert all(f"Description of {name}" not in system for name in {"alpha", "beta", "gamma"} - {permitted})
        results = tool_messages(model.inputs[after])
        assert [message.status for message in results] == ["success", "error"]
        assert results[0].content.endswith(f"PRIVATE-{permitted}-BODY")
        assert "PRIVATE-" not in results[1].content


async def test_filtered_views_share_one_stable_snapshot_and_refresh_on_rebuild(config, tmp_path, monkeypatch):
    alpha = put_skill(config, "alpha", "OLD-ALPHA")
    beta = put_skill(config, "beta", "OLD-BETA")
    config = config.model_copy(update={
        "agent": AgentConfig(skills=("alpha",)),
        "subagents": (config.subagents[0].model_copy(update={"skills": ("beta",)}),),
    })
    scans = []
    scan = SkillLoader._scan

    def count_scan(root):
        scans.append(root)
        return scan(root)

    monkeypatch.setattr(SkillLoader, "_scan", staticmethod(count_scan))
    worker = ScriptedModel([response(load("beta")), AIMessage(content="Checked")])
    parent = ScriptedModel([
        response(load("alpha")),
        response(call("delegate_task", {"agent_id": "worker", "query": "Load beta"}, "delegate")),
        AIMessage(content="Done"),
    ])
    graph = build_agent(config, workdir=str(tmp_path), model=parent, subagent_model=worker)
    assert len(scans) == 1
    alpha.write_text(alpha.read_text().replace("OLD-ALPHA", "NEW-ALPHA"))
    beta.write_text(beta.read_text().replace("OLD-BETA", "NEW-BETA"))
    put_skill(config, "gamma", "NEW-GAMMA")

    await graph.ainvoke({"messages": [HumanMessage(content="Load")]}, context=AgentContext(str(tmp_path)))

    assert len(scans) == 1
    assert tool_messages(parent.inputs[1])[0].content.endswith("OLD-ALPHA")
    assert tool_messages(worker.inputs[1])[0].content.endswith("OLD-BETA")
    refreshed = ScriptedModel([response(load("alpha"), load("gamma")), AIMessage(content="Done")])
    graph = build_agent(config, workdir=str(tmp_path), model=refreshed)
    await graph.ainvoke({"messages": [HumanMessage(content="Load")]}, context=AgentContext(str(tmp_path)))
    assert len(scans) == 2
    results = tool_messages(refreshed.inputs[-1])
    assert results[0].content.endswith("NEW-ALPHA") and results[1].status == "error"
    default = config.model_copy(update={"agent": AgentConfig()})
    components = build_agent_components(default, workdir=str(tmp_path))
    assert len(scans) == 3 and "Description of gamma" in components.skills_prompt
    alpha.unlink()
    with pytest.raises(ConfigError, match="Main agent.*alpha"):
        build_agent_components(config, workdir=str(tmp_path))


@pytest.mark.parametrize("interactive", [False, True])
def test_skill_cli_lists_full_library_despite_agent_selections(config, tmp_path, monkeypatch, interactive):
    for name in ("alpha", "beta"):
        put_skill(config, name, f"PRIVATE-{name}-BODY")
    path = tmp_path / "config.toml"
    path.write_text(path.read_text() + '\n[agent]\nskills = []\n')

    async def forbidden(*args, **kwargs):
        pytest.fail("Listing installed skills must not call the model")

    monkeypatch.setattr(cli, "ask", forbidden)
    args = ["--config", str(path), *([] if interactive else ["list-skill"])]
    result = CliRunner().invoke(cli.app, args, input="/skill\n/quit\n" if interactive else None)

    assert result.exit_code == 0, result.output
    assert "Installed skills" in result.stdout and "2 skills" in result.stdout
    assert all(f"Description of {name}" in result.stdout for name in ("alpha", "beta"))
    assert "PRIVATE-" not in result.output
    assert not config.runtime.sessions_dir.exists()


async def test_child_skill_selection_changes_require_a_new_session(config, tmp_path):
    for name in ("alpha", "beta"):
        put_skill(config, name, "Skill body")
    config = config.model_copy(update={
        "subagents": (config.subagents[0].model_copy(update={"skills": ("alpha",)}),),
    })
    session = create_session(config, workdir=str(tmp_path))
    changed = config.model_copy(update={
        "subagents": (config.subagents[0].model_copy(update={"skills": ("beta",)}),),
    })
    model = ScriptedModel([AIMessage(content="Must not run")])

    result = await ask(changed, session.session_id, "Hello", model=model)

    assert result.status == "failed" and result.error["type"] == "subagent_config_mismatch"
    assert "skills" in result.error["message"] and model.call_count == 0
