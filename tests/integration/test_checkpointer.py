from __future__ import annotations

import json
import subprocess
import sys

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rs_harness.agents.main import build_agent
from rs_harness.application import ask, create_session
from rs_harness.sessions.catalog import SessionCatalog
from rs_harness.sessions.checkpoints import open_checkpointer


class CaptureModelInput(AsyncCallbackHandler):
    def __init__(self):
        self.inputs = []

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        self.inputs.append([message.content for message in messages[0]])


async def checkpoint_messages(config, session):
    async with open_checkpointer(config) as saver:
        agent = build_agent(
            config, workdir=session.meta.workdir, checkpointer=saver,
            model=FakeListChatModel(responses=["unused"]),
        )
        state = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
        return [(message.type, message.content) for message in state.values["messages"]]


async def test_checkpoint_metadata_does_not_duplicate_session_identity(config):
    session = create_session(config)
    result = await ask(config, session.session_id, "hello", model=FakeListChatModel(responses=["hi"]))
    assert result.status == "succeeded", result.error
    async with open_checkpointer(config) as saver:
        agent = build_agent(
            config, workdir=session.meta.workdir, checkpointer=saver,
            model=FakeListChatModel(responses=["unused"]),
        )
        state = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
    assert state.config["configurable"]["thread_id"] == session.session_id
    assert "session_id" not in state.metadata
    assert state.metadata["interaction_id"] == result.interaction_id


async def test_same_session_can_resume_after_new_application_call(config):
    session = create_session(config)
    first = await ask(config, session.session_id, "first", model=FakeListChatModel(responses=["one"]))
    second = await ask(config, session.session_id, "second", model=FakeListChatModel(responses=["two"]))

    assert first.status == "succeeded", first.error
    assert second.status == "succeeded", second.error
    reopened = SessionCatalog.open(config, session.session_id)
    assert reopened.session_id == session.session_id
    events = [json.loads(line) for line in reopened.trajectory_path.read_text().splitlines()]
    starts = [event for event in events if event["event"] == "model_start"]
    assert len(starts) == 2
    assert starts[1]["payload"]["message_count"] > starts[0]["payload"]["message_count"]
    assert await checkpoint_messages(config, reopened) == [
        ("human", "first"), ("ai", "one"), ("human", "second"), ("ai", "two"),
    ]


async def test_different_sessions_have_different_contexts(config):
    first = create_session(config)
    second = create_session(config)
    assert first.session_id != second.session_id
    first_result = await ask(config, first.session_id, "first", model=FakeListChatModel(responses=["one"]))
    assert first_result.status == "succeeded", first_result.error
    result = await ask(config, second.session_id, "second", model=FakeListChatModel(responses=["two"]))
    assert result.status == "succeeded", result.error
    assert await checkpoint_messages(config, first) == [("human", "first"), ("ai", "one")]
    assert await checkpoint_messages(config, second) == [("human", "second"), ("ai", "two")]


@pytest.mark.parametrize("record_condition", ["missing", "tampered", "malformed", "unwritable"])
async def test_trajectory_cannot_supply_or_block_context(config, record_condition):
    session = create_session(config)
    first = await ask(config, session.session_id, "remember this", model=FakeListChatModel(responses=["remembered"]))
    assert first.status == "succeeded", first.error

    if record_condition in {"missing", "unwritable"}:
        session.trajectory_path.unlink()
        if record_condition == "unwritable":
            session.trajectory_path.mkdir()
    elif record_condition == "tampered":
        session.trajectory_path.write_text(json.dumps({
            "event": "user_input", "payload": {"text": "fabricated history"},
        }) + "\n")
    else:
        session.trajectory_path.write_text("invalid JSON\n")

    capture = CaptureModelInput()
    result = await ask(
        config, session.session_id, "follow up",
        model=FakeListChatModel(responses=["recalled"], callbacks=[capture]),
    )
    assert result.status == "succeeded", result.error
    assert len(capture.inputs) == 1
    assert capture.inputs[0][1:] == ["remember this", "remembered", "follow up"]
    assert await checkpoint_messages(config, session) == [
        ("human", "remember this"), ("ai", "remembered"),
        ("human", "follow up"), ("ai", "recalled"),
    ]
    assert bool(result.recording_errors) == (record_condition == "unwritable")


async def test_ask_returns_failure_for_missing_session_without_calling_model(config):
    model = FakeListChatModel(responses=["must not run"])
    result = await ask(config, "0" * 32, "hello", model=model)
    assert result.status == "failed"
    assert result.session_status == "unknown"
    assert result.error["type"] == "session_not_found"
    assert model.responses == ["must not run"]


async def test_ask_returns_failure_for_closed_session_without_writing_trajectory(config):
    session = create_session(config)
    SessionCatalog.close(config, session.session_id)
    result = await ask(config, session.session_id, "hello", model=FakeListChatModel(responses=["must not run"]))
    assert result.status == "failed"
    assert result.session_status == "closed"
    assert result.error["type"] == "session_closed"
    assert session.trajectory_path.read_text() == ""


async def test_ask_returns_failure_for_busy_session_without_writing_trajectory(config):
    session = create_session(config)
    with session.interaction_lock():
        result = await ask(config, session.session_id, "hello", model=FakeListChatModel(responses=["must not run"]))
    assert result.status == "failed"
    assert result.session_status == "running"
    assert result.error["type"] == "session_busy"
    assert session.trajectory_path.read_text() == ""


async def test_legacy_session_resumes_checkpoint_in_another_process(config, tmp_path):
    script = """
import asyncio
import sys
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from rs_harness.application import ask, create_session
from rs_harness.config import Config

async def run():
    config = Config.load(sys.argv[1])
    session_id = sys.argv[2] or create_session(config).session_id
    result = await ask(config, session_id, sys.argv[3], model=FakeListChatModel(responses=[sys.argv[4]]))
    print(result.model_dump_json())

asyncio.run(run())
"""

    def run_process(session_id, text, answer):
        process = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "config.toml"), session_id, text, answer],
            capture_output=True, text=True, timeout=15,
        )
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout)
        assert result["status"] == "succeeded", result
        return result

    first = run_process("", "first", "one")
    session = SessionCatalog.open(config, first["session_id"])
    earlier_trajectory = session.trajectory_path.read_bytes()

    second = run_process(session.session_id, "second", "two")
    assert first["session_id"] == second["session_id"]
    assert session.trajectory_path.read_bytes().startswith(earlier_trajectory)
    assert await checkpoint_messages(config, session) == [
        ("human", "first"), ("ai", "one"), ("human", "second"), ("ai", "two"),
    ]
