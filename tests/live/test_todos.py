import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rs_harness.agents.main import build_agent
from rs_harness.config import Config
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.live.test_tools_multimodal import ask_in_process, live_config_path
from tests.models import ScriptedModel

pytestmark = pytest.mark.live


async def snapshot(config_path, workdir, session_id):
    config = Config.load(config_path)
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={
        "sessions_dir": workdir / "sessions",
        "checkpoint_file": workdir / "sessions" / "checkpoints.sqlite3",
    })})
    async with open_checkpointer(config) as saver:
        agent = build_agent(
            config, workdir=str(workdir), checkpointer=saver,
            model=ScriptedModel([AIMessage(content="unused")]),
        )
        return await agent.aget_state({"configurable": {"thread_id": session_id}})


def tool_starts(workdir, session_id):
    trajectory = workdir / "sessions" / session_id / "trajectory.jsonl"
    return [record["payload"] for line in trajectory.read_text().splitlines()
            if (record := json.loads(line))["event"] == "tool_start"]


async def test_real_model_plans_updates_and_completes_work(live_config_path, tmp_path):
    result = ask_in_process(
        live_config_path, tmp_path,
        "Use write_todos to plan this task, keep the plan updated as you work, and mark it complete only after verification. "
        "First use write_file to create note.txt containing exactly 'alpha'. "
        "Then use edit_file to replace 'alpha' with 'beta'. Finally use read_file to verify the content. "
        "Wait for each dependent operation's result. Give the verified content in your final answer.",
    )
    state = await snapshot(live_config_path, tmp_path, result["session_id"])
    starts = tool_starts(tmp_path, result["session_id"])
    updates = [item["inputs"]["todos"] for item in starts if item["name"] == "write_todos"]

    assert (tmp_path / "note.txt").read_text() == "beta"
    assert "beta" in result["answer"]
    assert {item["name"] for item in starts} >= {"write_todos", "write_file", "edit_file", "read_file"}
    assert len(updates) >= 3
    assert any(item["status"] != "completed" for item in updates[0])
    assert any(
        any(item["status"] == "completed" for item in update)
        and any(item["status"] != "completed" for item in update)
        for update in updates[1:-1]
    )
    assert state.values["todos"] == updates[-1]
    assert state.values["todos"] and all(item["status"] == "completed" for item in state.values["todos"])


async def test_real_model_continues_saved_plan_in_another_process(live_config_path, tmp_path):
    first = ask_in_process(
        live_config_path, tmp_path,
        "Use write_todos to save a plan for these three steps: create resume.txt with exactly 'resume-token-7319'; "
        "replace 'resume-token' with 'verified-token' using edit_file; then read the file to verify the result. "
        "For this turn, only save the plan and tell me it is ready. Do not call file or shell tools yet. "
        "Keep the work unfinished until I ask you to continue.",
    )
    before = await snapshot(live_config_path, tmp_path, first["session_id"])
    assert before.values["todos"]
    assert any(item["status"] != "completed" for item in before.values["todos"])
    assert not (tmp_path / "resume.txt").exists()
    assert {item["name"] for item in tool_starts(tmp_path, first["session_id"])} == {"write_todos"}
    original_tool_ids = {message.tool_call_id for message in before.values["messages"] if isinstance(message, ToolMessage)}

    second = ask_in_process(
        live_config_path, tmp_path,
        "Continue the saved plan. Execute the remaining work, update the todos as you go, "
        "and report the verified final content after completing the plan.",
        session_id=first["session_id"],
    )
    after = await snapshot(live_config_path, tmp_path, first["session_id"])
    assert second["session_id"] == first["session_id"]
    assert second["interaction_id"] != first["interaction_id"]
    assert (tmp_path / "resume.txt").read_text() == "verified-token-7319"
    assert "verified-token-7319" in second["answer"]
    assert after.values["todos"] and all(item["status"] == "completed" for item in after.values["todos"])
    assert sum(isinstance(message, HumanMessage) for message in after.values["messages"]) == 2
    assert original_tool_ids < {message.tool_call_id for message in after.values["messages"] if isinstance(message, ToolMessage)}
