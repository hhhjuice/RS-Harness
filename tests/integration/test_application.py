from __future__ import annotations

import json

from tests.models import ToolCallingChatModel as FakeListChatModel

from rs_harness.application import ask, create_session


async def test_ask_writes_trajectory_and_returns_answer(config):
    session = create_session(config)
    model = FakeListChatModel(responses=["hello"])

    result = await ask(config, session.session_id, "hi", model=model)

    assert result.status == "succeeded", result.error
    assert result.answer == "hello"
    events = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert events[0]["event"] == "user_input"
    assert events[-1]["event"] == "interaction_succeeded"
    assert any(event["event"] == "model_start" for event in events)
    assert any(event["event"] == "model_end" for event in events)
    assert all(event["session_id"] == session.session_id for event in events)
    assert all("thread_id" not in event for event in events)
    assert config.api_key() not in session.trajectory_path.read_text()
    assert config.api_key() not in session.state_path.read_text()
