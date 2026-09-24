import base64
import hashlib
import json
import subprocess
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai.chat_models.base import _convert_message_to_dict

from rs_harness.agents.main import build_agent
from rs_harness.application import ask, create_session, resume
from rs_harness.inputs import build_human_message
from rs_harness.sessions.checkpoints import open_checkpointer
from rs_harness.tools import filesystem
from tests.images import write_test_image
from tests.models import ScriptedModel


def call(name, path, call_id):
    return {"name": name, "args": {"path": path}, "id": call_id, "type": "tool_call"}


async def test_image_batch_reaches_model_and_checkpoint_but_not_trajectory(config, tmp_path):
    data = write_test_image(tmp_path / "one.png")
    write_test_image(tmp_path / "two.png")
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            call("read_image", "one.png", "one"),
            {"name": "glob", "args": {"pattern": "*.png"}, "id": "glob", "type": "tool_call"},
            call("read_image", "missing.png", "missing"),
            call("read_image", "two.png", "two"),
        ]), AIMessage(content="done"),
    ])
    result = await ask(config, session.session_id, "look", model=model)
    assert result.status == "succeeded", result.error
    messages = model.inputs[-1]
    assert [message.type for message in messages] == ["system", "human", "ai", "tool", "tool", "tool", "tool", "human"]
    tools = {message.tool_call_id: message for message in messages if isinstance(message, ToolMessage)}
    assert set(tools) == {"one", "glob", "missing", "two"}
    assert tools["missing"].status == "error"
    assert all(isinstance(message.content, str) for message in tools.values())
    images = [block for block in messages[-1].content if block["type"] == "image"]
    assert len(images) == 2
    assert all(base64.b64decode(block["base64"]) == data for block in images)
    assert [block["text"] for block in messages[-1].content if block["type"] == "text"] == [
        "Image loaded from one.png", "Image loaded from two.png",
    ]
    wire = [_convert_message_to_dict(message) for message in messages]
    assert all(isinstance(message["content"], str) for message in wire if message["role"] == "tool")
    assert wire[-1]["role"] == "user"
    assert sum(block["type"] == "image_url" for block in wire[-1]["content"]) == 2

    async with open_checkpointer(config) as saver:
        agent = build_agent(config, workdir=str(tmp_path), model=model, checkpointer=saver)
        snapshot = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
    stored = snapshot.values["messages"]
    assert sum(isinstance(message, HumanMessage) for message in stored) == 1
    assert [message.content[1] for message in stored if isinstance(message, ToolMessage) and message.name == "read_image" and message.status == "success"] == images

    trajectory = session.trajectory_path.read_text()
    assert "base64" not in trajectory
    assert base64.b64encode(data).decode() not in trajectory
    events = [json.loads(line) for line in trajectory.splitlines()]
    outputs = [event["payload"]["output"] for event in events if event["event"] == "tool_end" and event["payload"]["output"]["name"] == "read_image"]
    assert {output["tool_call_id"] for output in outputs} == {"one", "two"}
    assert all(output["content"][1] == {
        "type": "image", "mime_type": "image/png", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    } for output in outputs)


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("kind", ["relative", "absolute", "symlink"])
async def test_external_image_requires_approval_before_reading(config, tmp_path, monkeypatch, decision, kind):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    image = tmp_path / "outside.png"
    write_test_image(image)
    path = "../outside.png" if kind == "relative" else str(image)
    if kind == "symlink":
        (workdir / "link.png").symlink_to(image)
        path = "link.png"
    reads = []
    original = filesystem.load_local_image

    def read(target):
        reads.append(target)
        return original(target)

    monkeypatch.setattr(filesystem, "load_local_image", read)
    session = create_session(config, workdir=str(workdir))
    tool_call = call("read_image", path, "image")
    tool_call["args"].update({"runtime": {"context": {"workdir": str(tmp_path)}}, "workdir": str(tmp_path), "approved": True})
    model = ScriptedModel([AIMessage(content="", tool_calls=[tool_call]), AIMessage(content="done")])
    pending = await ask(config, session.session_id, "look", model=model)
    assert pending.status == "interrupted", pending.error
    assert reads == []
    result = await resume(config, session.session_id, {
        pending.interrupts[0]["id"]: {"decisions": [{"type": decision}]},
    }, model=model)
    assert result.status == "succeeded", result.error
    assert reads == ([image] if decision == "approve" else [])
    message = next(message for message in model.inputs[-1] if isinstance(message, ToolMessage))
    assert message.tool_call_id == "image"
    assert message.status == ("success" if decision == "approve" else "error")
    if decision == "approve":
        assert str(image) in message.content


async def test_image_error_can_be_corrected_and_runtime_cannot_be_forged(config, tmp_path, monkeypatch):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    data = write_test_image(workdir / "image.png")
    session = create_session(config, workdir=str(workdir))
    monkeypatch.chdir(tmp_path.parent)
    fixed = call("read_image", "image.png", "fixed")
    fixed["args"].update({"runtime": {"context": {"workdir": "/"}}, "workdir": "/"})
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("read_image", 3, "invalid")]),
        AIMessage(content="", tool_calls=[call("read_image", "missing.png", "missing")]),
        AIMessage(content="", tool_calls=[fixed]), AIMessage(content="recovered"),
    ])
    result = await ask(config, session.session_id, "look", model=model)
    assert result.status == "succeeded", result.error
    tools = {message.tool_call_id: message for message in model.inputs[-1] if isinstance(message, ToolMessage)}
    assert tools["invalid"].status == tools["missing"].status == "error"
    assert tools["fixed"].status == "success"
    assert base64.b64decode(model.inputs[-1][-1].content[1]["base64"]) == data


@pytest.mark.parametrize("existing_images", [599, 600])
async def test_image_tool_obeys_whole_history_limits(config, tmp_path, existing_images):
    config = config.model_copy(update={"model": config.model.model_copy(update={
        "base_url": "https://api.deepseek.com", "model": "deepseek-flash",
    })})
    write_test_image(tmp_path / "image.png")
    session = create_session(config, workdir=str(tmp_path))
    message = build_human_message("look", images=["https://example.com/image.png"] * existing_images, config=config)
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("read_image", "image.png", "image")]), AIMessage(content="done"),
    ])
    result = await ask(config, session.session_id, message, model=model)
    assert result.status == ("succeeded" if existing_images == 599 else "failed"), result.error
    if existing_images == 600:
        assert result.error["type"] == "InputError"
        assert "too_many_images" in result.error["message"]
        assert model.call_count == 1
        followup = await ask(config, session.session_id, "again", model=model)
        assert followup.status == "failed"
        assert "too_many_images" in followup.error["message"]
        assert model.call_count == 1


def test_image_tool_history_recovers_in_new_process_without_source_or_trajectory(config, tmp_path):
    image = tmp_path / "image.png"
    data = write_test_image(image)
    script = """
import asyncio, json, sys
from langchain_core.messages import AIMessage
from rs_harness.application import ask, create_session
from rs_harness.config import Config
from tests.models import ScriptedModel

async def run():
    config = Config.load(sys.argv[1])
    session_id = sys.argv[2] or create_session(config, workdir=sys.argv[3]).session_id
    responses = [AIMessage(content='done')]
    if not sys.argv[2]:
        responses.insert(0, AIMessage(content='', tool_calls=[{'name':'read_image', 'args':{'path':'image.png'}, 'id':'image', 'type':'tool_call'}]))
    model = ScriptedModel(responses)
    result = await ask(config, session_id, 'follow up' if sys.argv[2] else 'look', model=model)
    print(json.dumps({'result': result.model_dump(), 'messages': [m.model_dump() for m in model.inputs[-1]]}))

asyncio.run(run())
"""

    def run(session_id):
        process = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "config.toml"), session_id, str(tmp_path)],
            capture_output=True, text=True, timeout=15,
        )
        assert process.returncode == 0, process.stderr
        payload = json.loads(process.stdout)
        assert payload["result"]["status"] == "succeeded", payload
        return payload

    first = run("")
    session_id = first["result"]["session_id"]
    image.unlink()
    (config.runtime.sessions_dir / session_id / "trajectory.jsonl").unlink()
    for _ in range(2):
        following = run(session_id)
        image_messages = [message for message in following["messages"] if message["type"] == "human" and isinstance(message["content"], list)]
        assert len(image_messages) == 1
        assert base64.b64decode(image_messages[0]["content"][1]["base64"]) == data
        assert following["messages"][-1]["content"] == "follow up"
