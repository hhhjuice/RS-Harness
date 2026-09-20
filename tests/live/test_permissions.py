import json
import subprocess
import sys

import pytest

from tests.live.test_tools_multimodal import live_config_path

pytestmark = pytest.mark.live


def permission_process(config_path, workdir, mode, payload="", session_id=""):
    script = '''
import asyncio
import json
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage, ToolMessage
from rs_harness.application import ask, create_session, get_pending_approvals, resume
from rs_harness.agents.main import build_agent
from rs_harness.config import Config
from rs_harness.sessions.checkpoints import open_checkpointer

async def run():
    config = Config.load(sys.argv[1])
    workdir = Path(sys.argv[2])
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={
        "sessions_dir": workdir / "sessions", "checkpoint_file": workdir / "sessions" / "checkpoints.sqlite3",
    })})
    session_id = sys.argv[3] or create_session(config, workdir=str(workdir)).session_id
    mode, payload = sys.argv[4:6]
    if mode == "ask":
        result = await ask(config, session_id, payload)
    elif mode == "query":
        result = await get_pending_approvals(config, session_id)
    else:
        result = await resume(config, session_id, json.loads(payload))
    async with open_checkpointer(config) as saver:
        agent = build_agent(config, workdir=str(workdir), checkpointer=saver)
        state = await agent.aget_state({"configurable":{"thread_id":session_id}})
    print(json.dumps({
        "result": result.model_dump(mode="json"),
        "humans": sum(isinstance(message, HumanMessage) for message in state.values["messages"]),
        "tools": [message.model_dump(mode="json") for message in state.values["messages"] if isinstance(message, ToolMessage)],
        "model": config.model.model,
    }))

asyncio.run(run())
'''
    process = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(workdir), session_id, mode, payload],
        capture_output=True, text=True, timeout=240,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def resume_expected(config_path, workdir, first, decision):
    pending = first["result"]
    queried = permission_process(config_path, workdir, "query", session_id=pending["session_id"])["result"]
    assert queried["interrupts"] == pending["interrupts"]
    assert queried["interaction_id"] == pending["interaction_id"]
    decisions = {pending["interrupts"][0]["id"]: {"decisions": [
        {"type": "approve"} if decision == "approve" else {"type": "reject", "message": "Keep the file unchanged. Do not retry or use another tool."},
    ]}}
    resumed = permission_process(config_path, workdir, "resume", json.dumps(decisions), pending["session_id"])
    assert resumed["result"]["status"] == "succeeded", resumed
    assert resumed["result"]["interaction_id"] == pending["interaction_id"]
    assert resumed["humans"] == first["humans"] == 1
    return resumed


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_real_model_dangerous_bash_across_processes(live_config_path, tmp_path, decision):
    target = tmp_path / "cleanup.txt"
    target.write_text("temporary test data")
    first = permission_process(
        live_config_path, tmp_path, "ask",
        "Call bash exactly once with command 'rm cleanup.txt' to delete this test file. "
        "Do not use other tools. After its result, report the outcome. If rejected, respect the rejection and do not retry.",
    )
    pending = first["result"]
    assert pending["status"] == "interrupted", first
    assert target.exists()
    assert len(pending["interrupts"]) == 1
    actions = pending["interrupts"][0]["action_requests"]
    assert len(actions) == 1
    assert actions[0]["name"] == "bash"
    assert actions[0]["args"] == {"command": "rm cleanup.txt"}
    assert first["tools"] == []
    resumed = resume_expected(live_config_path, tmp_path, first, decision)
    assert target.exists() == (decision == "reject")
    assert len(resumed["tools"]) == 1
    assert resumed["tools"][0]["status"] == ("success" if decision == "approve" else "error")


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("name", ["read_file", "write_file"])
def test_real_model_external_files(live_config_path, tmp_path, name, decision):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    external = tmp_path / "shared"
    external.mkdir()
    target = external / "note.txt"
    target.write_text("EXTERNAL-ORIGINAL-49261")
    instruction = "Read its contents." if name == "read_file" else "Set content to exactly 'EXTERNAL-CHANGED-49261'."
    first = permission_process(
        live_config_path, workdir, "ask",
        f"Call {name} exactly once with path '../shared/note.txt'. {instruction} "
        "Use no other tools. After the tool result, report the outcome. If rejected, respect the rejection and do not retry.",
    )
    pending = first["result"]
    assert pending["status"] == "interrupted", first
    assert len(pending["interrupts"]) == 1
    actions = pending["interrupts"][0]["action_requests"]
    assert len(actions) == 1
    assert actions[0]["name"] == name
    assert actions[0]["args"]["path"] == "../shared/note.txt"
    assert set(actions[0]["args"]) <= ({"path", "offset", "limit"} if name == "read_file" else {"path", "content"})
    if name == "write_file":
        assert actions[0]["args"]["content"] == "EXTERNAL-CHANGED-49261"
    assert target.read_text() == "EXTERNAL-ORIGINAL-49261"
    assert first["tools"] == []
    resumed = resume_expected(live_config_path, workdir, first, decision)
    assert len(resumed["tools"]) == 1
    message = resumed["tools"][0]
    assert message["status"] == ("success" if decision == "approve" else "error")
    if decision == "approve":
        assert json.loads(message["content"])["path"] == str(target.resolve())
    assert target.read_text() == ("EXTERNAL-CHANGED-49261" if name == "write_file" and decision == "approve" else "EXTERNAL-ORIGINAL-49261")
