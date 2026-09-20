import hashlib
import json
import subprocess
import sys

from langchain_core.messages import AIMessage, HumanMessage

from rs_harness.agents.main import build_agent
from rs_harness.application import ask, create_session
from rs_harness.inputs import build_human_message
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.images import write_test_image
from tests.models import ScriptedModel


async def test_user_input_trajectory_records_image_digest_instead_of_base64(config, tmp_path):
    data = write_test_image(tmp_path / "image.png")
    message = build_human_message("describe", images=[str(tmp_path / "image.png")])
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="ok")])

    assert (await ask(config, session.session_id, message, model=model)).status == "succeeded"

    trajectory = session.trajectory_path.read_text()
    assert "base64" not in trajectory
    events = [json.loads(line) for line in trajectory.splitlines()]
    user_input = next(event for event in events if event["event"] == "user_input")
    assert user_input["payload"]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image", "mime_type": "image/png", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
    ]
    followup = ScriptedModel([AIMessage(content="still ok")])
    assert (await ask(config, session.session_id, "again", model=followup)).status == "succeeded"
    assert followup.inputs[0][1].content == message.content


async def test_multimodal_checkpoint_survives_original_file_removal(config, tmp_path):
    image = tmp_path / "image.png"
    write_test_image(image)
    message = build_human_message("describe", images=[str(image), "https://example.com/sample.png"])
    session = create_session(config, workdir=str(tmp_path))
    model = ScriptedModel([AIMessage(content="red and blue")])
    assert (await ask(config, session.session_id, message, model=model)).status == "succeeded"
    image.unlink()
    session.trajectory_path.unlink()
    followup = ScriptedModel([AIMessage(content="still red and blue")])
    assert (await ask(config, session.session_id, "which is left?", model=followup)).status == "succeeded"
    assert followup.inputs[0][1].content == message.content
    assert len([entry for entry in followup.inputs[0] if isinstance(entry, HumanMessage)]) == 2
    async with open_checkpointer(config) as saver:
        agent = build_agent(config, workdir=str(tmp_path), model=followup, checkpointer=saver)
        snapshot = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
    assert snapshot.values["messages"][0].content == message.content
    assert len(snapshot.values["messages"]) == 4


def test_image_only_input_recovers_in_a_new_process(config, tmp_path):
    image = tmp_path / "image.png"
    write_test_image(image)
    script = """
import asyncio
import json
import sys
from langchain_core.messages import AIMessage
from rs_harness.application import ask, create_session
from rs_harness.config import Config
from rs_harness.inputs import build_human_message
from tests.models import ScriptedModel

async def run():
    config = Config.load(sys.argv[1])
    session_id = sys.argv[2] or create_session(config).session_id
    message = build_human_message(images=[sys.argv[3]]) if sys.argv[3] else 'follow up'
    model = ScriptedModel([AIMessage(content='done')])
    result = await ask(config, session_id, message, model=model)
    print(json.dumps({'result': result.model_dump(), 'history': [entry.content for entry in model.inputs[0]]}))

asyncio.run(run())
"""

    def run_process(session_id, image_path):
        process = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "config.toml"), session_id, image_path],
            capture_output=True, text=True, timeout=15,
        )
        assert process.returncode == 0, process.stderr
        payload = json.loads(process.stdout)
        assert payload["result"]["status"] == "succeeded", payload
        return payload

    first = run_process("", str(image))
    image.unlink()
    second = run_process(first["result"]["session_id"], "")
    assert second["history"][1] == first["history"][1]
    assert second["history"][-1] == "follow up"
    assert len(second["history"]) == 4
