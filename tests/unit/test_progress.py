import asyncio
import io
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult

from rs_harness.progress import CliProgress


def result(message):
    return LLMResult(generations=[[ChatGeneration(message=message)]])


async def emit(progress, run_id, text):
    await progress.on_llm_new_token(
        "", run_id=run_id,
        chunk=ChatGenerationChunk(message=AIMessageChunk(content="", additional_kwargs={"reasoning_content": text})),
    )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(("message", "expected"), [
    (AIMessage(content="answer", additional_kwargs={"reasoning_content": "思考\n\t下一步"}), "思考\n\t下一步\n"),
    (AIMessage(content="answer", additional_kwargs={"reasoning": "thinking\n"}), "thinking\n"),
    (AIMessage(content=[{"type": "reasoning", "reasoning": "thought"}, {"type": "text", "text": "answer"}],
               additional_kwargs={"reasoning_content": "thought"}), "thought\n"),
    (AIMessage(content=[{"type": "thinking", "thinking": "thought", "signature": "hidden"}]), "thought\n"),
    (AIMessage(content=[{"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary"}],
                        "encrypted_content": "hidden"}]), "summary\n"),
    (AIMessage(content="ordinary text"), ""),
    (AIMessage(content="ordinary text", additional_kwargs={"reasoning_content": None, "reasoning": {"private": "data"}}), ""),
    (AIMessage(content=[{"type": "reasoning", "summary": [], "encrypted_content": "hidden"}]), ""),
    (AIMessage(content=[{"type": "redacted_thinking", "data": "hidden"}]), ""),
])
async def test_only_reasoning_is_displayed_once(capsys, message, expected, streamed):
    progress = CliProgress()
    run_id = uuid4()
    await progress.on_chat_model_start({}, [[]], run_id=run_id)
    assert capsys.readouterr().err == ""
    if streamed:
        chunk = AIMessageChunk(content=message.content, additional_kwargs=message.additional_kwargs)
        await progress.on_llm_new_token("ordinary token", chunk=ChatGenerationChunk(message=chunk), run_id=run_id)
        await progress.on_llm_new_token("", chunk=ChatGenerationChunk(message=AIMessageChunk(content="")), run_id=run_id)
    await progress.on_llm_end(result(message), run_id=run_id)
    assert capsys.readouterr().err == ("[Thinking] " + expected if expected else "")
    assert not progress._model_runs and progress._reasoning_line is None


async def test_fragments_preserve_spacing_and_only_escape_terminal_controls(capsys):
    progress = CliProgress()
    run_id = uuid4()
    await progress.on_chat_model_start({}, [[]], run_id=run_id)
    text = "中文" * 200 + "\n\t next \x1b[31m\r\u202e"
    for fragment in (text[:1], text[1:400], text[400:]):
        await emit(progress, run_id, fragment)
    # A complete response must not replay streamed content, even if different.
    await progress.on_llm_end(result(AIMessage(content="answer", additional_kwargs={"reasoning_content": "do not replay"})), run_id=run_id)
    assert capsys.readouterr().err == "[Thinking] " + "中文" * 200 + "\n\t next \\u001b[31m\\u000d\\u202e\n"


async def test_child_stream_and_complete_response_are_hidden(capsys):
    progress = CliProgress()
    delegate, child_chain, child_model, parent_model = (uuid4() for _ in range(4))
    await progress.on_tool_start({"name": "delegate_task"}, "", run_id=delegate)
    await progress.on_chain_start({}, {}, run_id=child_chain, parent_run_id=delegate)
    await progress.on_chat_model_start({}, [[]], run_id=child_model, parent_run_id=child_chain)
    await emit(progress, child_model, "hidden stream")
    await progress.on_llm_end(result(AIMessage(content="", additional_kwargs={"reasoning_content": "hidden response"})), run_id=child_model)
    await progress.on_chain_end({}, run_id=child_chain, parent_run_id=delegate)
    await progress.on_tool_end("hidden tool result", run_id=delegate)
    await progress.on_chat_model_start({}, [[]], run_id=parent_model)
    await emit(progress, parent_model, "visible parent")
    await progress.on_llm_end(result(AIMessage(content="answer")), run_id=parent_model)
    output = capsys.readouterr().err
    assert "hidden" not in output and output.endswith("[Thinking] visible parent\n")
    assert output.count("[Thinking]") == 1
    assert not progress._hidden_runs and not progress._model_runs


@pytest.mark.parametrize("error", [RuntimeError("failed"), asyncio.CancelledError()])
@pytest.mark.parametrize("root_cleanup", [False, True])
async def test_failed_or_cancelled_stream_finishes_line_and_allows_next_run(capsys, error, root_cleanup):
    progress = CliProgress()
    run_id = uuid4()
    await progress.on_chat_model_start({}, [[]], run_id=run_id)
    await emit(progress, run_id, "partial thought")
    if root_cleanup:
        await progress.on_chain_error(error, run_id=uuid4())
    else:
        await progress.on_llm_error(error, run_id=run_id)
    assert not progress._model_runs and progress._reasoning_line is None
    next_id = uuid4()
    await progress.on_chat_model_start({}, [[]], run_id=next_id)
    await emit(progress, next_id, "new thought")
    await progress.on_llm_end(result(AIMessage(content="done")), run_id=next_id)
    assert capsys.readouterr().err == "[Thinking] partial thought\n[Thinking] new thought\n"


@pytest.mark.parametrize("failure", ["write", "flush"])
async def test_terminal_failure_does_not_replay_reasoning(monkeypatch, failure):
    class BrokenStream(io.StringIO):
        def write(self, value):
            if failure == "write":
                raise OSError("unavailable")
            return super().write(value)

        def flush(self):
            if failure == "flush":
                raise ValueError("closed")
            return super().flush()

    progress = CliProgress()
    run_id = uuid4()
    await progress.on_chat_model_start({}, [[]], run_id=run_id)
    monkeypatch.setattr("rs_harness.progress.sys.stderr", BrokenStream())
    await emit(progress, run_id, "consumed")
    recovered = io.StringIO()
    monkeypatch.setattr("rs_harness.progress.sys.stderr", recovered)
    await progress.on_llm_end(result(AIMessage(content="done", additional_kwargs={"reasoning_content": "consumed"})), run_id=run_id)
    assert recovered.getvalue() == "\n"
    assert not progress._model_runs and progress._reasoning_line is None


async def test_tool_notice_starts_on_a_new_line(capsys):
    progress = CliProgress()
    run_id, tool_id = uuid4(), uuid4()
    await progress.on_chat_model_start({}, [[]], run_id=run_id)
    await emit(progress, run_id, "thought")
    await progress.on_tool_start({"name": "glob"}, "", run_id=tool_id)
    await progress.on_llm_end(result(AIMessage(content="")), run_id=run_id)
    assert capsys.readouterr().err == "[Thinking] thought\n[Tool] Running glob\n"
