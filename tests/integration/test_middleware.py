from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, NotRequired

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from typer.testing import CliRunner

from rs_harness import cli
from rs_harness.agents.main import AgentContext, build_agent
from rs_harness.agents.middleware import StopHookMiddleware, build_middleware
from rs_harness.application import ask, create_session
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.models import ScriptedModel


class ProbeState(AgentState):
    probe_count: NotRequired[int]


@dataclass(frozen=True)
class ProbeContext:
    workdir: str


TOOL_CALLS: list[str] = []


@tool
def echo(text: str, runtime: ToolRuntime[ProbeContext]) -> str:
    """Return text while recording the runtime context."""

    TOOL_CALLS.append(runtime.context.workdir)
    return text


class ProbeMiddleware(AgentMiddleware[ProbeState, ProbeContext]):
    state_schema = ProbeState

    def __init__(self, label: str, events: list[str]):
        super().__init__()
        self.label = label
        self.events = events

    @property
    def name(self) -> str:
        return self.label

    async def abefore_agent(self, state, runtime):
        self.events.append(f"{self.label}:before_agent:{runtime.context.workdir}")

    async def abefore_model(self, state, runtime):
        self.events.append(f"{self.label}:before_model")

    async def aafter_model(self, state, runtime):
        self.events.append(f"{self.label}:after_model")
        if self.label == "A":
            return {"probe_count": state.get("probe_count", 0) + 1}
        return None

    async def aafter_agent(self, state, runtime):
        self.events.append(f"{self.label}:after_agent")

    async def awrap_model_call(self, request, handler):
        self.events.append(f"{self.label}:model_in:{request.runtime.context.workdir}")
        response = await handler(request)
        self.events.append(f"{self.label}:model_out")
        return response

    async def awrap_tool_call(self, request, handler):
        call_id = request.tool_call["id"]
        self.events.append(f"{self.label}:tool_in:{call_id}:{request.runtime.context.workdir}")
        response = await handler(request)
        self.events.append(f"{self.label}:tool_out:{call_id}")
        return response


def _agent(model, *, middleware, tools=(echo,), checkpointer=None):
    return create_agent(
        model=model,
        tools=list(tools),
        middleware=list(middleware),
        context_schema=ProbeContext,
        checkpointer=checkpointer,
        system_prompt="base prompt",
    )


@pytest.mark.asyncio
async def test_async_hooks_state_and_context_order():
    events: list[str] = []
    TOOL_CALLS.clear()
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "echo", "args": {"text": "one"}, "id": "call-1", "type": "tool_call"},
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    saver = InMemorySaver()
    agent = _agent(
        model,
        middleware=[
            ModelCallLimitMiddleware(run_limit=2, exit_behavior="error"),
            ProbeMiddleware("A", events),
            ProbeMiddleware("B", events),
        ],
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "hook-order"}}

    result = await agent.ainvoke(
        {"messages": [HumanMessage("run") ]},
        config=config,
        context=ProbeContext("/workspace"),
    )

    assert result["messages"][-1].content == "done"
    assert result["probe_count"] == 2
    assert TOOL_CALLS == ["/workspace"]
    assert events == [
        "A:before_agent:/workspace",
        "B:before_agent:/workspace",
        "A:before_model",
        "B:before_model",
        "A:model_in:/workspace",
        "B:model_in:/workspace",
        "B:model_out",
        "A:model_out",
        "B:after_model",
        "A:after_model",
        "A:tool_in:call-1:/workspace",
        "B:tool_in:call-1:/workspace",
        "B:tool_out:call-1",
        "A:tool_out:call-1",
        "A:before_model",
        "B:before_model",
        "A:model_in:/workspace",
        "B:model_in:/workspace",
        "B:model_out",
        "A:model_out",
        "B:after_model",
        "A:after_model",
        "B:after_agent",
        "A:after_agent",
    ]

    snapshot = await agent.aget_state(config)
    assert snapshot.values["probe_count"] == 2
    assert "context" not in snapshot.values
    assert set(echo.args) == {"text"}


@pytest.mark.asyncio
async def test_multiple_tool_calls_are_all_processed_without_order_assumption():
    events: list[str] = []
    TOOL_CALLS.clear()
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "echo", "args": {"text": "one"}, "id": "call-a", "type": "tool_call"},
                    {"name": "echo", "args": {"text": "two"}, "id": "call-b", "type": "tool_call"},
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = _agent(model, middleware=[ProbeMiddleware("A", events)], checkpointer=InMemorySaver())

    result = await agent.ainvoke(
        {"messages": [HumanMessage("run") ]},
        config={"configurable": {"thread_id": "multiple-tools"}},
        context=ProbeContext("/workspace"),
    )

    assert result["messages"][-1].content == "done"
    assert TOOL_CALLS == ["/workspace", "/workspace"]
    assert {event.split(":")[2] for event in events if event.startswith("A:tool_in:")} == {
        "call-a",
        "call-b",
    }
    assert model.call_count == 2
    assert {
        message.tool_call_id: message.content
        for message in result["messages"] if message.type == "tool"
    } == {"call-a": "one", "call-b": "two"}


class OverrideMiddleware(AgentMiddleware[ProbeState, ProbeContext]):
    state_schema = ProbeState

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(system_message=SystemMessage(content="temporary")))


@pytest.mark.asyncio
async def test_model_request_override_is_per_call(config):
    model = ScriptedModel([AIMessage(content="done"), AIMessage(content="follow-up answer")])
    run_config = {"configurable": {"thread_id": "override"}}

    async with open_checkpointer(config) as saver:
        agent = _agent(model, middleware=[OverrideMiddleware()], tools=(), checkpointer=saver)
        await agent.ainvoke(
            {"messages": [HumanMessage("run")]},
            config=run_config,
            context=ProbeContext("/workspace"),
        )

    assert model.inputs[0][0].content == "temporary"
    assert model.inputs[0][-1].content == "run"

    async with open_checkpointer(config) as saver:
        agent = _agent(model, middleware=[], tools=(), checkpointer=saver)
        snapshot = await agent.aget_state(run_config)
        assert [(message.type, message.content) for message in snapshot.values["messages"]] == [
            ("human", "run"), ("ai", "done"),
        ]
        await agent.ainvoke(
            {"messages": [HumanMessage("follow up")]},
            config=run_config,
            context=ProbeContext("/workspace"),
        )

    assert [message.content for message in model.inputs[1]] == [
        "base prompt", "run", "done", "follow up",
    ]


@pytest.mark.asyncio
async def test_middleware_state_survives_sqlite_reopen(config):
    config = config.model_copy(update={
        "runtime": config.runtime.model_copy(update={"max_model_calls": 1}),
    })
    run_config = {"configurable": {"thread_id": "persisted-probe"}}
    model = ScriptedModel([AIMessage(content="done") for _ in range(3)])

    async with open_checkpointer(config) as saver:
        agent = _agent(
            model, tools=(), checkpointer=saver,
            middleware=[*build_middleware(config), ProbeMiddleware("A", [])],
        )
        await agent.ainvoke(
            {"messages": [HumanMessage("first")]},
            config=run_config, context=ProbeContext("/workspace"),
        )

    async with open_checkpointer(config) as saver:
        agent = _agent(
            model, tools=(), checkpointer=saver,
            middleware=[*build_middleware(config), ProbeMiddleware("A", [])],
        )
        snapshot = await agent.aget_state(run_config)
        assert snapshot.values["probe_count"] == 1
        result = await agent.ainvoke(
            {"messages": [HumanMessage("second")]},
            config=run_config, context=ProbeContext("/workspace"),
        )
        assert result["probe_count"] == 2
        assert [(message.type, message.content) for message in result["messages"]] == [
            ("human", "first"), ("ai", "done"), ("human", "second"), ("ai", "done"),
        ]
        isolated = await agent.ainvoke(
            {"messages": [HumanMessage("separate")]},
            config={"configurable": {"thread_id": "isolated-probe"}},
            context=ProbeContext("/other-workspace"),
        )
        assert isolated["probe_count"] == 1
        assert [message.content for message in isolated["messages"]] == ["separate", "done"]

    assert model.call_count == 3


@pytest.mark.asyncio
async def test_model_call_limit_blocks_next_call_and_resets_for_new_run():
    TOOL_CALLS.clear()
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "echo", "args": {"text": "one"}, "id": "limited", "type": "tool_call"},
                ],
            ),
            AIMessage(content="after-limit"),
        ]
    )
    config = {"configurable": {"thread_id": "limit"}}
    agent = _agent(
        model,
        middleware=[ModelCallLimitMiddleware(run_limit=1, exit_behavior="error")],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(Exception, match=r"run limit \(1/1\)") as error:
        await agent.ainvoke(
            {"messages": [HumanMessage("run")]},
            config=config,
            context=ProbeContext("/limited"),
        )
    assert type(error.value).__name__ == "ModelCallLimitExceededError"
    assert model.call_count == 1
    assert TOOL_CALLS == ["/limited"]

    result = await agent.ainvoke(
        {"messages": [HumanMessage("retry") ]},
        config=config,
        context=ProbeContext("/limited"),
    )
    assert result["messages"][-1].content == "after-limit"
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_model_call_limit_allows_exact_budget():
    model = ScriptedModel([AIMessage(content="done")])
    agent = _agent(
        model,
        middleware=[ModelCallLimitMiddleware(run_limit=1, exit_behavior="error")],
        tools=(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("run")]},
        config={"configurable": {"thread_id": "exact-budget"}},
        context=ProbeContext("/workspace"),
    )

    assert result["messages"][-1].content == "done"
    assert model.call_count == 1


def test_production_agent_factory_wires_middleware_and_context(config, monkeypatch):
    captured: dict[str, Any] = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)
    result = build_agent(
        config,
        workdir="/workspace",
        checkpointer="checkpoint",
        model=ScriptedModel([AIMessage(content="unused")]),
    )

    assert result is not None
    assert captured["context_schema"] is AgentContext
    assert captured["checkpointer"] == "checkpoint"
    assert {tool.name for tool in captured["tools"]} == {"read_file", "write_file", "edit_file", "glob", "bash"}
    assert len(captured["middleware"]) == 7
    assert isinstance(captured["middleware"][0], HumanInTheLoopMiddleware)
    assert isinstance(captured["middleware"][1], ModelCallLimitMiddleware)
    assert captured["middleware"][1].run_limit == config.runtime.max_model_calls
    assert captured["middleware"][1].exit_behavior == "error"
    assert isinstance(captured["middleware"][-2], TodoListMiddleware)
    assert isinstance(captured["middleware"][-1], StopHookMiddleware)
    assert captured["middleware"][-1].hooks == ()


class ApplicationContextCapture(AgentMiddleware[AgentState, AgentContext]):
    def __init__(self):
        super().__init__()
        self.captured: list[str] = []

    async def abefore_agent(self, state, runtime):
        self.captured.append(runtime.context.workdir)


@pytest.mark.asyncio
async def test_application_passes_session_workdir_as_runtime_context(config, monkeypatch, tmp_path):
    workdir = tmp_path / "session-workdir"
    workdir.mkdir()
    capture = ApplicationContextCapture()

    def fake_build_agent(config, *, workdir, checkpointer=None, model=None):
        return create_agent(
            model=model,
            tools=[],
            middleware=[capture],
            context_schema=AgentContext,
            checkpointer=checkpointer,
        )

    monkeypatch.setattr("rs_harness.application.build_agent", fake_build_agent)
    session = create_session(config, workdir=str(workdir))
    monkeypatch.chdir(tmp_path)

    result = await ask(
        config,
        session.session_id,
        "hello",
        model=ScriptedModel([AIMessage(content="done")]),
    )

    assert result.status == "succeeded", result.error
    assert capture.captured == [str(workdir.resolve())]


@pytest.mark.asyncio
async def test_application_records_budget_failure_and_allows_next_interaction(config, monkeypatch):
    config = config.model_copy(update={
        "runtime": config.runtime.model_copy(update={"max_model_calls": 1}),
    })
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "echo", "args": {"text": "one"}, "id": "limited", "type": "tool_call"},
                ],
            ),
            AIMessage(content="recovered"),
        ]
    )

    def fake_build_agent(config, *, workdir, checkpointer=None, model=None):
        return _agent(
            model,
            middleware=build_middleware(config),
            checkpointer=checkpointer,
        )

    monkeypatch.setattr("rs_harness.application.build_agent", fake_build_agent)
    session = create_session(config)

    failed = await ask(config, session.session_id, "first", model=model)

    assert failed.status == "failed"
    assert failed.answer is None
    assert failed.error["type"] == "ModelCallLimitExceededError"
    events = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "user_input", "model_start", "model_end", "tool_start", "tool_end", "interaction_failed",
    ]
    assert model.call_count == 1

    recovered = await ask(config, session.session_id, "second", model=model)

    assert recovered.status == "succeeded", recovered.error
    assert recovered.answer == "recovered"
    assert model.call_count == 2


@pytest.mark.parametrize("interactive", [False, True])
def test_cli_budget_failure_and_next_interaction(config, monkeypatch, tmp_path, interactive):
    config = config.model_copy(update={
        "runtime": config.runtime.model_copy(update={"max_model_calls": 1}),
    })
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"text": "one"}, "id": "cli-limit", "type": "tool_call"},
        ]),
        AIMessage(content="recovered"),
    ])
    session = create_session(config)

    def build_test_agent(loaded_config, *, workdir, checkpointer=None, model=None):
        return _agent(model, middleware=build_middleware(loaded_config), checkpointer=checkpointer)

    async def ask_with_model(loaded_config, session_id, text, **kwargs):
        return await ask(loaded_config, session_id, text, model=model, **kwargs)

    monkeypatch.setattr("rs_harness.application.build_agent", build_test_agent)
    monkeypatch.setattr(cli, "ask", ask_with_model)
    monkeypatch.setattr(cli, "_load", lambda path: config)
    arguments = ["--config", str(tmp_path / "config.toml"), "--session-id", session.session_id]
    runner = CliRunner()

    if interactive:
        result = runner.invoke(cli.app, arguments, input="first\nsecond\n/quit\n")
        assert result.exit_code == 0, result.output
        assert "ModelCallLimitExceededError" in result.stderr
        assert "recovered" in result.stdout
    else:
        failed = runner.invoke(cli.app, ["ask", "first", *arguments, "--json"])
        assert failed.exit_code == 1, failed.output
        payload = json.loads(failed.stdout)
        assert payload["status"] == "failed"
        assert payload["answer"] is None
        assert payload["error"]["type"] == "ModelCallLimitExceededError"
        assert failed.stderr == ""
        recovered = runner.invoke(cli.app, ["ask", "second", *arguments, "--json"])
        assert recovered.exit_code == 0, recovered.output
        assert json.loads(recovered.stdout)["answer"] == "recovered"

    events = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "user_input", "model_start", "model_end", "tool_start", "tool_end", "interaction_failed",
        "user_input", "model_start", "model_end", "interaction_succeeded",
    ]
    assert model.call_count == 2
