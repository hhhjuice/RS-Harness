import asyncio
import io
import json

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.tools import tool

from rs_harness import application, cli
from rs_harness.config import Config
from rs_harness.context_usage import model_key
from rs_harness.models import ReasoningChatOpenAI
from rs_harness.progress import CliProgress
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.integration.test_progress import RecordingStream, install_agent


def event(delta=None, *, finish=None, usage=None):
    payload = {
        "id": "test-response", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        payload["usage"] = usage
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


USAGE = {"prompt_tokens": 37, "completion_tokens": 11, "total_tokens": 48}


class PausedStream(httpx.AsyncByteStream):
    def __init__(self, tail, entered, release):
        self.tail = tail
        self.entered = entered
        self.release = release
        self.closed = False

    async def __aiter__(self):
        yield event({"role": "assistant", "reasoning_content": "先检查"})
        # The callback has consumed the first chunk before the next read.
        self.entered.set()
        await self.release.wait()
        yield self.tail

    async def aclose(self):
        self.closed = True


def make_model(client, *, streaming=True):
    return ReasoningChatOpenAI(
        model="test-model", api_key="offline-test-key", base_url="https://example.invalid/v1",
        # Only the async client is exercised; avoid constructing an unused sync transport.
        streaming=streaming, stream_usage=True, client=object(), http_async_client=client, max_retries=0,
    )


@pytest.mark.parametrize("stream_usage", [None, True, False])
async def test_production_stream_usage_calibrates_context(config, tmp_path, monkeypatch, stream_usage):
    config_path = tmp_path / "config.toml"
    settings = 'base_url = "https://example.invalid/v1"\n'
    if stream_usage is not None:
        settings += f"stream_usage = {json.dumps(stream_usage)}\n"
    config_path.write_text(config_path.read_text().replace("[model]\n", "[model]\n" + settings))
    config = Config.load(config_path)
    requests = []

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        wire = event({"role": "assistant", "content": "answer"}) + event({}, finish="stop")
        if payload.get("stream_options", {}).get("include_usage"):
            wire += event(usage=USAGE)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire + b"data: [DONE]\n\n")

    session = application.create_session(config, workdir=str(tmp_path))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        def production_model(**kwargs):
            return ReasoningChatOpenAI(**kwargs, client=object(), http_async_client=client, max_retries=0)

        monkeypatch.setattr("rs_harness.models.ReasoningChatOpenAI", production_model)
        response = await application.ask(config, session.session_id, "question")

    assert response.status == "succeeded" and response.answer == "answer", response.error
    expected_usage = stream_usage is not False
    assert len(requests) == 1 and requests[0]["stream"] is True
    assert requests[0].get("stream_options") == ({"include_usage": True} if expected_usage else None)
    async with open_checkpointer(config, read_only=True) as saver:
        saved = await saver.aget_tuple({"configurable": {"thread_id": session.session_id, "checkpoint_ns": ""}})
    state = saved.checkpoint["channel_values"]
    usage = await application.get_context_usage(config, session.session_id)
    if expected_usage:
        assert state["messages"][-1].usage_metadata["input_tokens"] == USAGE["prompt_tokens"]
        assert state["context_calibration"]["input_tokens"] == USAGE["prompt_tokens"]
        assert state["context_calibration"]["model_key"] == model_key(config)
        assert usage.estimation_method == "calibrated"
    else:
        assert state["messages"][-1].usage_metadata is None
        assert state["context_calibration"] is None
        assert usage.estimation_method == "approximate"


async def test_reasoning_arrives_before_model_completion_and_tools_execute_once(config, tmp_path, monkeypatch, capsys):
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []
    first = PausedStream(
        event({"reasoning_content": "文件\n"})
        + event({"content": "ordinary preamble", "tool_calls": [{
            "index": 0, "id": "write", "type": "function",
            "function": {"name": "write_file", "arguments": '{"path":"note.txt",'},
        }]})
        + event({"tool_calls": [{"index": 0, "function": {"arguments": '"content":"saved"}'}}]})
        + event({}, finish="tool_calls"),
        entered, release,
    )

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["stream"] is True
        ending = event(usage=USAGE) if payload.get("stream_options", {}).get("include_usage") else b""
        ending += b"data: [DONE]\n\n"
        if len(requests) == 1:
            first.tail += ending
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=first)
        wire = (
            event({"role": "assistant", "reasoning": "核对结果"})
            + event({"content": "保存"}) + event({"content": "完成"})
            + event({}, finish="stop") + ending
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire)

    class Capture(AsyncCallbackHandler):
        def __init__(self):
            self.messages = []

        async def on_llm_end(self, response, **kwargs):
            self.messages.append(response.generations[0][0].message)

    output = io.StringIO()
    progress, capture = CliProgress(), Capture()
    session = application.create_session(config, workdir=str(tmp_path))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = make_model(client)
        with monkeypatch.context() as patch:
            patch.setattr("rs_harness.progress.sys.stderr", output)
            task = asyncio.create_task(application.ask(config, session.session_id, "save", model=model, callbacks=[progress, capture]))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert output.getvalue() == "[Thinking] 先检查"
                assert not task.done() and not (tmp_path / "note.txt").exists()
                release.set()
                response = await asyncio.wait_for(task, 10)
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    assert response.status == "succeeded", response.error
    assert response.answer == "保存完成"
    assert (tmp_path / "note.txt").read_text() == "saved"
    assert len(requests) == 2 and first.closed
    text = output.getvalue()
    assert text.startswith("[Thinking] 先检查文件\n[Tool] Running write_file ")
    assert text.endswith("[Thinking] 核对结果\n") and text.count("核对结果") == 1
    assert text.count("[Thinking]") == 2
    assert text.count("[Tool] Running write_file ") == text.count("[Tool] Completed write_file (") == 1
    assert not any(value in text for value in ("[Model]", "ordinary preamble", "保存完成"))
    assert capture.messages[0].additional_kwargs["reasoning_content"] == "先检查文件\n"
    assert capture.messages[1].additional_kwargs["reasoning"] == "核对结果"
    assert capture.messages[0].tool_calls[0]["args"] == {"path": "note.txt", "content": "saved"}
    assert all(message.usage_metadata["input_tokens"] == 37 for message in capture.messages)
    records = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert sum(row["event"] == "model_end" for row in records) == 2
    assert sum(row["event"] == "tool_start" for row in records) == 1
    assert not progress._model_runs and progress._reasoning_line is None
    cli._print_result(response, False, progress)
    assert capsys.readouterr().out == "保存完成\n"


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("value", ["returned thought", "", None, {"encrypted": "hidden"}])
async def test_complete_http_response_preserves_only_string_reasoning(capsys, key, value):
    async def handler(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(200, json={
            "id": "test-response", "object": "chat.completion", "created": 1, "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer", key: value}, "finish_reason": "stop"}],
            "usage": USAGE,
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await make_model(client, streaming=False).ainvoke("question", config={"callbacks": [CliProgress()]})
    assert response.content == "answer" and response.usage_metadata["input_tokens"] == 37
    assert response.additional_kwargs.get(key) == (value if isinstance(value, str) else None)
    assert capsys.readouterr().err == ("[Thinking] " + value + "\n" if isinstance(value, str) and value else "")


@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_stream_closes_and_clears_progress(config, tmp_path, monkeypatch, cancel):
    entered, release = asyncio.Event(), asyncio.Event()

    class InterruptedStream(PausedStream):
        async def __aiter__(self):
            async for chunk in super().__aiter__():
                yield chunk
            raise httpx.ReadError("stream interrupted")

    stream = InterruptedStream(b"", entered, release)

    async def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    output = io.StringIO()
    monkeypatch.setattr("rs_harness.progress.sys.stderr", output)
    session = application.create_session(config, workdir=str(tmp_path))
    progress = CliProgress()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(application.ask(config, session.session_id, "run", model=make_model(client), callbacks=[progress]))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                release.set()
                response = await asyncio.wait_for(task, 5)
                assert response.status == "failed"
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert output.getvalue().startswith("[Thinking] 先检查\n")
    assert output.getvalue().count("先检查") == 1
    assert stream.closed and not progress._model_runs and progress._reasoning_line is None


async def test_failed_stream_output_does_not_repeat_a_tool(config, tmp_path, monkeypatch):
    calls, requests = [], []
    monkeypatch.setattr("rs_harness.progress.sys.stderr", RecordingStream(fail=True))

    @tool
    async def touch() -> str:
        """Perform one observable action."""
        calls.append("touch")
        return "done"

    install_agent(monkeypatch, [touch])

    async def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            delta = {"tool_calls": [{"index": 0, "id": "touch", "type": "function", "function": {"name": "touch", "arguments": "{}"}}]}
        else:
            delta = {"content": "finished"}
        wire = event({"role": "assistant", "reasoning_content": "thought"}) + event(delta)
        wire += event({}, finish="tool_calls" if len(requests) == 1 else "stop") + b"data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire)

    session = application.create_session(config, workdir=str(tmp_path))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await application.ask(config, session.session_id, "touch", model=make_model(client), callbacks=[CliProgress()])
    assert response.status == "succeeded" and response.answer == "finished", response.error
    assert calls == ["touch"] and len(requests) == 2
