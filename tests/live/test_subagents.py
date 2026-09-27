import json

import pytest
import tomlkit
from langchain_core.messages import AIMessage, ToolMessage

from rs_harness.config import Config
from rs_harness.sessions.checkpoints import open_checkpointer

from tests.live.test_permissions import permission_process, resume_expected
from tests.live.test_tools_multimodal import ask_in_process, live_config_path

pytestmark = pytest.mark.live


def trace(workdir, session_id):
    return [json.loads(line) for line in
            (workdir / "sessions" / session_id / "trajectory.jsonl").read_text().splitlines()]


def starts(workdir, session_id):
    return {row["run_id"]: row for row in trace(workdir, session_id) if row["event"].endswith("_start")}


def is_delegated(row, events, *, run_id=None):
    while row["parent_run_id"]:
        row = events[row["parent_run_id"]]
        if (row["event"] == "tool_start" and row["payload"]["name"] == "delegate_task"
                and (run_id is None or row["run_id"] == run_id)):
            return True
    return False


def test_real_model_delegates_independent_file_checks(live_config_path, tmp_path):
    (tmp_path / "alpha.txt").write_text("alpha-value-7319")
    (tmp_path / "beta.txt").write_text("beta-value-2846")
    result = ask_in_process(
        live_config_path, tmp_path,
        "Use delegate_task with agent_id engineer twice: one independent task to read alpha.txt and report its exact content, "
        "and another to read beta.txt and report its exact content. Give each engineer the relative path. "
        "After both return, report both contents yourself. Do not read the files in the main agent.",
    )
    assert "alpha-value-7319" in result["answer"] and "beta-value-2846" in result["answer"]
    events = starts(tmp_path, result["session_id"])
    tools = [row for row in events.values() if row["event"] == "tool_start"]
    assert sum(row["payload"]["name"] == "delegate_task" for row in tools) == 2
    reads = [row for row in tools if row["payload"]["name"] == "read_file"]
    assert {row["payload"]["inputs"]["path"] for row in reads} == {"alpha.txt", "beta.txt"}
    assert all(is_delegated(row, events) for row in reads)


def test_real_model_engineer_edits_and_parent_verifies(live_config_path, tmp_path):
    result = ask_in_process(
        live_config_path, tmp_path,
        "Delegate this entire task in one delegate_task call with agent_id engineer: use write_file to create result.txt "
        "with exactly 'alpha-7319', then edit_file to replace 'alpha' with 'beta', and read_file to "
        "verify it. After the engineer returns, you as the main agent must call read_file on result.txt "
        "yourself and report the verified final content. Wait for each dependent result.",
    )
    assert (tmp_path / "result.txt").read_text() == "beta-7319"
    assert "beta-7319" in result["answer"]
    events = starts(tmp_path, result["session_id"])
    tools = [row for row in events.values() if row["event"] == "tool_start"]
    assert {row["payload"]["name"] for row in tools if is_delegated(row, events)} >= {"write_file", "edit_file", "read_file"}
    assert any(row["payload"]["name"] == "read_file" and not is_delegated(row, events) for row in tools)


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_real_model_parent_approval_after_child_denial(persistent_live_config_path, tmp_path, decision):
    live_config_path = persistent_live_config_path
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    target = tmp_path / "external.txt"
    target.write_text("original")
    first = permission_process(
        live_config_path, workdir, "ask",
        "Call delegate_task with agent_id engineer exactly once to do this task: call write_file exactly once with "
        "path '../external.txt' and content 'approved-7319'. Tell the engineer to use no other tools, "
        "respect rejection, and never retry or bypass a rejection. For this permission test, ask it "
        "to make that one tool call even though denial is expected, then report the denial. "
        "After the engineer returns, you as the main agent must call write_file once yourself with "
        "the same path and content, using your own permission flow. After your result, report it; "
        "if your action is rejected, do not retry or use another tool.",
    )
    pending = first["result"]
    assert pending["status"] == "interrupted", first
    assert target.read_text() == "original"
    assert len(pending["interrupts"]) == 1
    assert len(first["tools"]) == 1 and first["tools"][0]["name"] == "delegate_task"
    assert first["tools"][0]["content"]
    actions = pending["interrupts"][0]["action_requests"]
    assert len(actions) == 1 and actions[0]["name"] == "write_file"
    assert actions[0]["args"] == {"path": "../external.txt", "content": "approved-7319"}
    events = starts(workdir, pending["session_id"])
    assert any(row["event"] == "tool_start" and row["payload"]["name"] == "delegate_task" for row in events.values())
    resumed = resume_expected(live_config_path, workdir, first, decision)
    assert target.read_text() == ("approved-7319" if decision == "approve" else "original")
    assert resumed["result"]["answer"]
    events = starts(workdir, pending["session_id"])
    writes = [row for row in events.values() if row["event"] == "tool_start" and row["payload"]["name"] == "write_file"]
    assert len(writes) == int(decision == "approve")
    assert all(not is_delegated(row, events) for row in writes)
    followup = ask_in_process(
        live_config_path, workdir,
        "Ask the same engineer through delegate_task to report the path and intended content from "
        "its prior task and why its own write was denied. Tell it to use only its "
        "existing conversation and no tools. Do not provide those details in the delegated query. "
        "Return the engineer's answer without other actions.",
        session_id=pending["session_id"],
    )
    assert "external.txt" in followup["answer"] and "approved-7319" in followup["answer"]
    assert target.read_text() == ("approved-7319" if decision == "approve" else "original")


@pytest.fixture
def persistent_live_config_path(live_config_path, tmp_path):
    config = Config.load(live_config_path)
    document = tomlkit.parse(live_config_path.read_text())
    for field in ("sessions_dir", "checkpoint_file", "system_prompt_file"):
        document["runtime"][field] = str(getattr(config.runtime, field))
    for definition, item in zip(config.subagents, document["subagents"], strict=True):
        item["system_prompt_file"] = str(definition.system_prompt_file)
        if definition.id == "engineer":
            item["persistence"] = "per_thread"
    path = tmp_path / "live.toml"
    path.touch(mode=0o600)
    path.write_text(tomlkit.dumps(document))
    return path


def test_real_model_researcher_scope_and_retained_private_observation(live_config_path, tmp_path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("Internal reference: ORBIT-7391-BLUE. Review completed on day 17.")
    first = ask_in_process(
        live_config_path, tmp_path,
        "Choose the configured workspace research agent by its role and tools, and delegate this task: "
        "read evidence.txt once and remember all details for a later question. Its final response must "
        "be only 'inspection complete', with no file content, reference, or date. Do not use tools "
        "in the main agent except delegate_task. Reply only 'inspection complete' after it returns.",
    )
    events = starts(tmp_path, first["session_id"])
    tools = [row for row in events.values() if row["event"] == "tool_start"]
    dispatches = [row for row in tools if row["payload"]["name"] == "delegate_task"]
    assert len(dispatches) == 1 and dispatches[0]["payload"]["inputs"]["agent_id"] == "researcher"
    assert "ORBIT-7391-BLUE" not in first["answer"]
    dispatch_result = next(row["payload"]["output"] for row in trace(tmp_path, first["session_id"])
                           if row["event"] == "tool_end" and row["run_id"] == dispatches[0]["run_id"])
    assert "ORBIT-7391-BLUE" not in str(dispatch_result)
    child_tools = [row for row in tools if is_delegated(row, events)]
    assert {row["payload"]["name"] for row in child_tools} <= {"read_file", "glob", "read_image", "write_todos"}
    assert any(row["payload"]["name"] == "read_file" for row in child_tools)
    evidence.unlink()
    second = ask_in_process(
        live_config_path, tmp_path,
        "Ask the same researcher to recall the internal reference in the previously inspected file. "
        "The file is gone. Use only the researcher's existing conversation, no new file reads or other "
        "tools. The main agent must get the answer from delegate_task and report it.",
        session_id=first["session_id"],
    )
    assert "ORBIT-7391-BLUE" in second["answer"]
    queries = [row["payload"]["inputs"]["query"] for row in trace(tmp_path, first["session_id"])
               if row["event"] == "tool_start" and row["payload"]["name"] == "delegate_task"]
    assert all("ORBIT-7391-BLUE" not in query for query in queries)
    events = starts(tmp_path, first["session_id"])
    reads = [row for row in events.values() if row["event"] == "tool_start" and row["payload"]["name"] == "read_file"]
    assert len(reads) == 1 and is_delegated(reads[0], events)


def test_real_model_invocation_mode_starts_without_prior_observation(live_config_path, tmp_path):
    evidence = tmp_path / "ephemeral.txt"
    evidence.write_text("Private code: EPHEMERAL-8631")
    first = ask_in_process(
        live_config_path, tmp_path,
        "Call delegate_task using engineer to read ephemeral.txt. Tell it to return only 'checked', "
        "without copying the code. Do not use other main-agent tools and reply only 'checked'.",
    )
    assert "EPHEMERAL-8631" not in first["answer"]
    events = starts(tmp_path, first["session_id"])
    dispatch_ids = {row["run_id"] for row in events.values()
                    if row["event"] == "tool_start" and row["payload"]["name"] == "delegate_task"}
    assert len(dispatch_ids) == 1
    assert all("EPHEMERAL-8631" not in str(row["payload"]["output"])
               for row in trace(tmp_path, first["session_id"])
               if row["event"] == "tool_end" and row["run_id"] in dispatch_ids)
    evidence.unlink()
    second = ask_in_process(
        live_config_path, tmp_path,
        "Call delegate_task again using engineer. Ask it if its current conversation contains the "
        "previous file's private code. Do not provide the code, read any files, or use other tools. "
        "If unavailable, the engineer must state that clearly rather than guess. Report its answer.",
        session_id=first["session_id"],
    )
    assert "EPHEMERAL-8631" not in second["answer"]
    events = starts(tmp_path, first["session_id"])
    dispatches = [row for row in events.values()
                  if row["event"] == "tool_start" and row["payload"]["name"] == "delegate_task"]
    assert len(dispatches) == 2
    assert all("EPHEMERAL-8631" not in row["payload"]["inputs"]["query"] for row in dispatches)
    assert any(row["event"] == "tool_start" and row["payload"]["name"] == "read_file"
               and is_delegated(row, events, run_id=dispatches[0]["run_id"]) for row in events.values())
    followup_events = [row for row in events.values() if is_delegated(row, events, run_id=dispatches[1]["run_id"])]
    # A fresh child receives only its system prompt and this query, with no previous messages.
    assert [row["payload"]["message_count"] for row in followup_events if row["event"] == "model_start"] == [2]
    assert not any(row["event"] == "tool_start" for row in followup_events)


async def test_real_model_child_denial_returns_control_to_parent(live_config_path, tmp_path):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("original")
    result = ask_in_process(
        live_config_path, workdir,
        "For a permission check, delegate exactly once to engineer. Ask it to call write_file once "
        "with path '../external.txt' and content 'must-not-write', even though denial is expected, "
        "then report the result without retrying or bypassing the denial. After it reports denial, "
        "you as the main agent must write recovery.txt inside the workspace with exactly "
        "'handled-by-parent'. Do not attempt the external write yourself. Report the outcome.",
    )
    assert external.read_text() == "original"
    assert (workdir / "recovery.txt").read_text() == "handled-by-parent"
    assert not result["interrupts"]
    events = starts(workdir, result["session_id"])
    tools = [row for row in events.values() if row["event"] == "tool_start"]
    assert sum(row["payload"]["name"] == "delegate_task" for row in tools) == 1
    writes = [row for row in tools if row["payload"]["name"] == "write_file"]
    assert len(writes) == 1 and writes[0]["payload"]["inputs"]["path"] == "recovery.txt"
    assert not is_delegated(writes[0], events)
    config = Config.load(live_config_path)
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={
        "sessions_dir": workdir / "sessions", "checkpoint_file": workdir / "sessions/checkpoints.sqlite3",
    })})
    async with open_checkpointer(config) as saver:
        checkpoints = [item async for item in saver.alist({"configurable": {"thread_id": result["session_id"]}})]
    child_messages = [message for item in checkpoints if item.config["configurable"]["checkpoint_ns"]
                      for message in item.checkpoint["channel_values"].get("messages", [])]
    attempts = {item["id"] for message in child_messages if isinstance(message, AIMessage)
                for item in message.tool_calls if item["name"] == "write_file" and item["args"].get("path") == "../external.txt"}
    assert len(attempts) == 1
    assert any(isinstance(message, ToolMessage) and message.tool_call_id in attempts
               and message.status == "error" and "Permission denied" in message.content for message in child_messages)
