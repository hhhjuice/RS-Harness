from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphBubbleUp

from rs_harness import application, cli
from rs_harness.agents.main import AgentContext
from rs_harness.agents.middleware import StopHookMiddleware, build_middleware
from rs_harness.progress import CliProgress
from tests.models import ScriptedModel


def call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def install_agent(monkeypatch, tools):
    def build_agent(loaded_config, *, workdir, checkpointer=None, model=None, subagents=None):
        return create_agent(
            model=model,
            tools=list(tools),
            middleware=build_middleware(loaded_config, tools=loaded_config.agent.tools),
            context_schema=AgentContext,
            checkpointer=checkpointer,
        )

    monkeypatch.setattr(application, "build_agent", build_agent)


class RecordingStream(io.StringIO):
    def __init__(self, *, fail: bool = False):
        super().__init__()
        self.fail = fail

    def write(self, value: str) -> int:
        if self.fail:
            raise OSError("console unavailable")
        return super().write(value)

    def flush(self) -> None:
        if self.fail:
            raise OSError("console unavailable")

    def text(self) -> str:
        return self.getvalue()


@pytest.mark.asyncio
async def test_tool_start_is_reported_before_side_effect_and_completion_after(config, tmp_path, monkeypatch):
    stream = RecordingStream()
    monkeypatch.setattr("rs_harness.progress.sys.stderr", stream)
    observed: list[str] = []

    @tool("touch")
    async def touch(path: str) -> str:
        """Write one marker file."""
        observed.append(stream.text())
        Path(path).write_text("done", encoding="utf-8")
        return "done"

    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("touch", {"path": str(tmp_path / "marker")}, "touch")]),
        AIMessage(content="finished"),
    ])
    install_agent(monkeypatch, [touch])
    session = application.create_session(config, workdir=str(tmp_path))

    result = await application.ask(config, session.session_id, "touch it", model=model, callbacks=[CliProgress()])

    assert result.status == "succeeded", result.error
    assert (tmp_path / "marker").read_text(encoding="utf-8") == "done"
    assert observed and "[Tool] Running touch " in observed[0]
    assert "[Tool] Completed" not in observed[0]
    output = stream.text()
    assert output.count("[Tool] Running touch ") == 1
    assert "[Tool] Running touch (" not in output
    assert output.count("[Tool] Completed touch (") == 1
    assert output.index("[Tool] Running touch ") < output.index("[Tool] Completed touch (")


@pytest.mark.asyncio
async def test_parallel_tools_report_completion_in_completion_order(config, tmp_path, monkeypatch):
    stream = RecordingStream()
    monkeypatch.setattr("rs_harness.progress.sys.stderr", stream)
    fast_finished = asyncio.Event()
    slow_started = asyncio.Event()

    class CompletionProgress(CliProgress):
        async def on_tool_end(self, output, **kwargs):
            await super().on_tool_end(output, **kwargs)
            if output.tool_call_id == "fast":
                fast_finished.set()

    @tool("inspect")
    async def inspect(path: str) -> str:
        """Inspect a path, allowing a later call to finish first."""
        if path == "slow.txt":
            slow_started.set()
            await asyncio.wait_for(fast_finished.wait(), timeout=5)
        else:
            await asyncio.wait_for(slow_started.wait(), timeout=5)
        return path

    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            call("inspect", {"path": "slow.txt"}, "slow"),
            call("inspect", {"path": "fast.txt"}, "fast"),
        ]),
        AIMessage(content="finished"),
    ])
    install_agent(monkeypatch, [inspect])
    session = application.create_session(config, workdir=str(tmp_path))

    result = await application.ask(config, session.session_id, "run both", model=model, callbacks=[CompletionProgress()])

    assert result.status == "succeeded", result.error
    output = stream.text()
    starts = [line for line in output.splitlines() if line.startswith("[Tool] Running inspect ")]
    assert len(starts) == 2 and all("inspect (" not in line for line in starts)
    completions = [line for line in output.splitlines() if line.startswith("[Tool] Completed inspect (")]
    assert len(completions) == 2
    for line, path in zip(completions, ("fast.txt", "slow.txt"), strict=True):
        assert line.startswith("[Tool] Completed inspect (")
        assert line.endswith(f') {{"path":"{path}"}}')


@pytest.mark.asyncio
async def test_denied_and_approved_tools_report_real_execution_only(config, tmp_path, monkeypatch):
    denied_stream = RecordingStream()
    monkeypatch.setattr("rs_harness.progress.sys.stderr", denied_stream)
    target = tmp_path / "target.txt"
    target.write_text("keep", encoding="utf-8")
    denied_model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("bash", {"command": f"sudo rm {target}"}, "deny")]),
        AIMessage(content="denied"),
    ])
    session = application.create_session(config, workdir=str(tmp_path))
    denied = await application.ask(config, session.session_id, "deny it", model=denied_model, callbacks=[CliProgress()])
    assert denied.status == "succeeded", denied.error
    assert target.exists()
    assert "blocked" in denied_stream.text()
    assert "[Tool]" not in denied_stream.text()

    approved_stream = RecordingStream()
    monkeypatch.setattr("rs_harness.progress.sys.stderr", approved_stream)
    approved_model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("bash", {"command": f"rm {target}"}, "approve")]),
        AIMessage(content="approved"),
    ])
    approved_session = application.create_session(config, workdir=str(tmp_path))
    pending = await application.ask(config, approved_session.session_id, "approve it", model=approved_model, callbacks=[CliProgress()])
    assert pending.status == "interrupted", pending.error
    assert "[Tool]" not in approved_stream.text()
    decision = {pending.interrupts[0]["id"]: {"decisions": [{"type": "approve"}]}}
    resumed = await application.resume(config, approved_session.session_id, decision, model=approved_model, callbacks=[CliProgress()])
    assert resumed.status == "succeeded", resumed.error
    assert not target.exists()
    assert "[Tool] Running bash " in approved_stream.text()
    assert "[Tool] Completed bash (" in approved_stream.text()


@pytest.mark.asyncio
@pytest.mark.parametrize("todos, expected", [
    (
        [
            {"content": "Inspect input", "status": "completed"},
            {"content": "Modify target", "status": "in_progress"},
            {"content": "Run checks", "status": "pending"},
            {"content": "Summarize result", "status": "pending"},
            {"content": "Clean temporary files", "status": "pending"},
        ],
        True,
    ),
    ([], True),
    ([{"content": "invalid", "status": "done"}], False),
])
async def test_todo_progress_only_shows_successful_updates(config, tmp_path, todos, expected, monkeypatch):
    stream = RecordingStream()
    monkeypatch.setattr("rs_harness.progress.sys.stderr", stream)
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("write_todos", {"todos": todos}, "todos")]),
        AIMessage(content="finished"),
    ])
    session = application.create_session(config, workdir=str(tmp_path))

    result = await application.ask(config, session.session_id, "plan", model=model, callbacks=[CliProgress()])

    assert result.status == "succeeded", result.error
    output = stream.text()
    assert "[Tool]" not in output
    if expected and todos:
        assert output.count("[Todo] Updated") == 1
        assert all(todo["content"] in output for todo in todos)
        assert "remaining" not in output.lower()
    elif expected:
        assert output.count("[Todo] Plan cleared") == 1
    else:
        assert output.count("[Todo] Update failed") == 1
        assert "Updated" not in output
        assert "invalid" not in output


@pytest.mark.asyncio
async def test_progress_output_failure_does_not_repeat_side_effect(config, tmp_path, monkeypatch):
    monkeypatch.setattr("rs_harness.progress.sys.stderr", RecordingStream(fail=True))
    calls = 0

    @tool("touch")
    async def touch(path: str) -> str:
        """Write one marker file."""
        nonlocal calls
        calls += 1
        Path(path).write_text("done", encoding="utf-8")
        return "done"

    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("touch", {"path": str(tmp_path / "marker")}, "touch")]),
        AIMessage(content="finished"),
    ])
    install_agent(monkeypatch, [touch])
    session = application.create_session(config, workdir=str(tmp_path))

    result = await application.ask(config, session.session_id, "touch it", model=model, callbacks=[CliProgress()])

    assert result.status == "succeeded", result.error
    assert calls == 1
    assert (tmp_path / "marker").read_text(encoding="utf-8") == "done"


async def test_stop_hook_continuation_reports_action_without_message_content(config, tmp_path, monkeypatch, capsys):
    def check(messages):
        return "private continuation instruction" if messages[-1].content == "draft" else None

    def with_stop_hook(config, **kwargs):
        middleware = build_middleware(config, **kwargs)
        middleware[-1] = StopHookMiddleware((check,))
        return middleware

    monkeypatch.setattr("rs_harness.agents.main.build_middleware", with_stop_hook)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(
        config, session.session_id, "finish", callbacks=[CliProgress()],
        model=ScriptedModel([AIMessage(content="draft"), AIMessage(content="finished")]),
    )

    assert result.status == "succeeded", result.error
    assert result.answer == "finished"
    output = capsys.readouterr().err
    assert "StopHookMiddleware.after_agent · continuing model execution" in output
    assert "private continuation instruction" not in output
    assert "draft" not in output


@pytest.mark.parametrize("with_image", [False, True])
async def test_routine_middleware_is_silent(config, tmp_path, capsys, with_image):
    session = application.create_session(config, workdir=str(tmp_path))
    message = HumanMessage(content=[{"type": "image", "url": "https://example.com/image.png"}]) if with_image else "hello"
    result = await application.ask(
        config, session.session_id, message, callbacks=[CliProgress()],
        model=ScriptedModel([AIMessage(content="done")]),
    )

    assert result.status == "succeeded", result.error
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("failure", ["budget", "image", "stop", "tool"])
async def test_fatal_error_reason_is_reported_once(config, tmp_path, monkeypatch, capsys, failure):
    message = "run"
    model = ScriptedModel([AIMessage(content="done")])
    if failure == "budget":
        config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"max_model_calls": 1})})
        model = ScriptedModel([AIMessage(content="", tool_calls=[call("glob", {"pattern": "*.txt"}, "glob")])])
        marker = "Model call budget exhausted; stopping further calls"
    elif failure == "image":
        message = HumanMessage(content=[{"type": "image", "url": "https://example.com/image.png"}])
        model = ScriptedModel([AIMessage(content="unreachable")], profile={"image_inputs": False})
        marker = "image_inputs_not_supported"
    elif failure == "stop":
        def check(messages):
            raise RuntimeError("stop check failed")

        def with_stop_hook(config, **kwargs):
            middleware = build_middleware(config, **kwargs)
            middleware[-1] = StopHookMiddleware((check,))
            return middleware

        monkeypatch.setattr("rs_harness.agents.main.build_middleware", with_stop_hook)
        marker = "stop check failed"
    else:
        @tool
        def fail() -> str:
            """Raise an unexpected tool error."""
            raise RuntimeError("tool crashed")

        install_agent(monkeypatch, [fail])
        model = ScriptedModel([AIMessage(content="", tool_calls=[call("fail", {}, "fail")])])
        marker = "tool crashed"

    session = application.create_session(config, workdir=str(tmp_path))
    progress = CliProgress()
    result = await application.ask(config, session.session_id, message, model=model, callbacks=[progress])
    cli._print_result(result, False, progress)

    assert result.status == "failed", result.error
    output = capsys.readouterr().err
    assert output.count(marker) == 1
    assert output.count(result.error["type"]) == 1
    assert output.count("This run failed.") == 1


@pytest.mark.parametrize("error_message", [False, True])
async def test_handled_tool_error_has_one_notice_and_agent_can_continue(config, tmp_path, capsys, monkeypatch, error_message):
    if error_message:
        @tool("read_file")
        def unavailable(path: str) -> ToolMessage:
            """Return an error observation without raising."""
            return ToolMessage(content=f"{path} unavailable", tool_call_id="read", status="error")

        install_agent(monkeypatch, [unavailable])
    session = application.create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("read_file", {"path": "missing.txt"}, "read")]),
        AIMessage(content="recovered"),
    ])
    progress = CliProgress()
    result = await application.ask(config, session.session_id, "read", model=model, callbacks=[progress])
    cli._print_result(result, False, progress)

    assert result.status == "succeeded", result.error
    output = capsys.readouterr()
    assert output.out == "recovered\n"
    assert output.err.count("[Tool] Running read_file ") == 1
    assert "[Tool] Completed read_file" not in output.err
    assert output.err.index("[Tool] Running read_file ") < output.err.index("[Tool] Failed read_file (")
    assert output.err.count("Failed") == 1
    assert output.err.count("ToolError" if error_message else "ToolException") == 1
    assert "missing.txt" in output.err
    failure = next(line for line in output.err.splitlines() if line.startswith("[Tool] Failed read_file ("))
    assert ') {"path":"missing.txt"} (' in failure
    assert "ToolErrorMiddleware" not in output.err


async def test_duplicate_plan_calls_have_one_rejection_notice(config, tmp_path, capsys):
    session = application.create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("write_todos", {"todos": []}, name) for name in ("one", "two")]),
        AIMessage(content="done"),
    ])
    result = await application.ask(config, session.session_id, "plan", model=model, callbacks=[CliProgress()])

    assert result.status == "succeeded", result.error
    output = capsys.readouterr().err
    assert output.count("[Todo] Update rejected") == 1
    assert "[Tool]" not in output
    assert "Plan cleared" not in output
    assert "Updated" not in output


@pytest.mark.parametrize("signal", [asyncio.CancelledError, GraphBubbleUp])
@pytest.mark.parametrize("stage", ["tool", "stop"])
async def test_control_flow_is_not_reported_as_failure(config, tmp_path, monkeypatch, capsys, signal, stage):
    entered = asyncio.Event()

    async def interrupt():
        entered.set()
        if signal is asyncio.CancelledError:
            await asyncio.Event().wait()
        else:
            raise signal()

    model = ScriptedModel([AIMessage(content="done")])
    if stage == "tool":
        @tool
        async def abort() -> str:
            """Stop execution through a control-flow signal."""
            await interrupt()

        install_agent(monkeypatch, [abort])
        model = ScriptedModel([AIMessage(content="", tool_calls=[call("abort", {}, "abort")])])
    else:
        async def check(messages):
            await interrupt()

        def with_stop_hook(config, **kwargs):
            middleware = build_middleware(config, **kwargs)
            middleware[-1] = StopHookMiddleware((check,))
            return middleware

        monkeypatch.setattr("rs_harness.agents.main.build_middleware", with_stop_hook)
    session = application.create_session(config, workdir=str(tmp_path))

    task = asyncio.create_task(application.ask(config, session.session_id, "stop", model=model, callbacks=[CliProgress()]))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if signal is asyncio.CancelledError:
            task.cancel()
        with pytest.raises(signal):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    output = capsys.readouterr().err
    assert "failed" not in output
    assert output.count("[Tool] Running abort") == int(stage == "tool")
    assert "[Tool] Completed" not in output
    assert "[Middleware]" not in output


@pytest.mark.parametrize("feedback", [None, ""])
async def test_stop_hook_without_continuation_is_silent(config, tmp_path, monkeypatch, capsys, feedback):
    def with_stop_hook(config, **kwargs):
        middleware = build_middleware(config, **kwargs)
        middleware[-1] = StopHookMiddleware((lambda messages: feedback,))
        return middleware

    monkeypatch.setattr("rs_harness.agents.main.build_middleware", with_stop_hook)
    session = application.create_session(config, workdir=str(tmp_path))
    result = await application.ask(
        config, session.session_id, "finish", callbacks=[CliProgress()],
        model=ScriptedModel([AIMessage(content="done")]),
    )

    assert result.status == "succeeded", result.error
    assert capsys.readouterr().err == ""
