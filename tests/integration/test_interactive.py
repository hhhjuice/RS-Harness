import asyncio
import json

import pytest
from tests.models import ToolCallingChatModel as FakeListChatModel
from pydantic import PrivateAttr
from typer.testing import CliRunner

from rs_harness import application, cli
from rs_harness.sessions.catalog import SessionCatalog


class LoopBoundChatModel(FakeListChatModel):
    _owner_loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)
    _inputs: list[list[str]] = PrivateAttr(default_factory=list)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if self._owner_loop is None:
            self._owner_loop = asyncio.get_running_loop()
        self._owner_loop.call_soon(lambda: None)
        self._inputs.append([message.content for message in messages])
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


@pytest.mark.parametrize("resume", [False, True])
def test_interactive_multiturn_preserves_client_loop_and_history(config, tmp_path, monkeypatch, resume):
    arguments = ["--config", str(tmp_path / "config.toml")]
    if resume:
        session = application.create_session(config)
        result = asyncio.run(application.ask(
            config, session.session_id, "earlier message",
            model=FakeListChatModel(responses=["earlier answer"]),
        ))
        assert result.status == "succeeded", result.error
        arguments.extend(["--session-id", session.session_id])

    model = LoopBoundChatModel(responses=["first answer", "second answer"])

    async def ask_with_model(loaded_config, session_id, text, **kwargs):
        return await application.ask(loaded_config, session_id, text, model=model, **kwargs)

    monkeypatch.setattr(cli, "ask", ask_with_model)
    result = CliRunner().invoke(
        cli.app, arguments, input="first question\nsecond question\n/quit\n"
    )

    assert result.exit_code == 0, result.output
    assert "Error:" not in result.output, result.output
    assert "first answer" in result.output
    assert "second answer" in result.output
    assert len(model._inputs) == 2
    assert model._inputs[1][-3:] == ["first question", "first answer", "second question"]
    if resume:
        assert model._inputs[0][1:3] == ["earlier message", "earlier answer"]
    assert model._owner_loop.is_closed()

    sessions = SessionCatalog.list(config)
    assert len(sessions) == 1
    assert sessions[0].meta.status == "running"
    events = [json.loads(line) for line in sessions[0].trajectory_path.read_text().splitlines()]
    completed = [event for event in events if event["event"] == "interaction_succeeded"]
    assert len(completed) == (3 if resume else 2)
    assert completed[-1]["interaction_id"] != completed[-2]["interaction_id"]
