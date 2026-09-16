import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from rs_harness.runtime import Session
from rs_harness.agent_loop import InputRequest, close_session, run
from rs_harness.config import ConfigError


def make_config(tmp_path: Path):
    from rs_harness.config import Config

    prompt = tmp_path / "system.md"
    prompt.write_text(
        "system\nWorking directory: {workdir}\n"
        "Only create, modify, or edit files inside this directory.",
        encoding="utf-8",
    )
    return Config(
        model={"model": "test-model", "api_key_env": "UNUSED_RS_KEY"},
        runtime={
            "sessions_dir": tmp_path / "sessions",
            "system_prompt_file": prompt,
            "dotenv_file": tmp_path / "missing.env",
        },
    )


def test_create_open_and_append_complete_message_ledger(tmp_path):
    config = make_config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = Session.create(config, workdir=workspace)
    assert session.state.status == "running"
    assert session.messages() == [
        {
            "role": "system",
            "content": (
                f"system\nWorking directory: {workspace.resolve()}\n"
                "Only create, modify, or edit files inside this directory."
            ),
        }
    ]

    with session.locked():
        session.append({"role": "user", "content": "hello"})
        session.append(
            {"role": "assistant", "content": "hi", "tool_calls": [{"id": "c1"}]}
        )
    reopened = Session.open(config, session.state.session_id)
    assert reopened.messages()[-2:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "c1"}]},
    ]

    records = [
        json.loads(line)
        for line in (session.path / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["seq"] for record in records] == [1, 2, 3]
    assert all(set(record) == {"seq", "timestamp", "message"} for record in records)
    assert not (session.path / "events.jsonl").exists()


def test_close_is_idempotent_and_rejects_new_messages(tmp_path):
    config = make_config(tmp_path)
    session = Session.create(config)
    first = session.close(status="closed", answer="done", stop_reason="user_closed")
    second = session.close(status="closed", answer="ignored")
    assert second == first
    assert not (session.path / "result.json").exists()
    with session.locked(), pytest.raises(ValueError, match="session_closed"):
        session.append({"role": "user", "content": "late"})


def test_ledger_redacts_credentials_and_data_values(tmp_path):
    session = Session.create(make_config(tmp_path))
    with session.locked():
        session.append(
            {
                "role": "tool",
                "content": {
                    "token": "secret-token",
                    "payload": "data:image/tiff;base64," + "A" * 100,
                    "text": "ordinary content",
                },
            }
        )
    raw = (session.path / "messages.jsonl").read_text(encoding="utf-8")
    assert "secret-token" not in raw
    assert "data:image/tiff" not in raw
    assert "ordinary content" in raw


def test_close_derives_last_assistant_text(tmp_path):
    session = Session.create(make_config(tmp_path))
    with session.locked():
        session.append({"role": "assistant", "content": "derived answer"})
    result = session.close()
    assert result.answer == "derived answer"
    assert session.state.answer == "derived answer"


def test_close_derives_text_blocks_and_explicit_answer_wins(tmp_path):
    session = Session.create(make_config(tmp_path))
    with session.locked():
        session.append(
            {"role": "assistant", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}
        )
    assert session.close().answer == "one\ntwo"

    explicit = Session.create(make_config(tmp_path))
    with explicit.locked():
        explicit.append({"role": "assistant", "content": "ledger answer"})
    assert explicit.close(answer="explicit answer").answer == "explicit answer"


def test_close_does_not_derive_text_from_assistant_tool_planning(tmp_path):
    session = Session.create(make_config(tmp_path))
    with session.locked():
        session.append(
            {
                "role": "assistant",
                "content": "planning text",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
            }
        )
    assert session.close().answer is None


def test_close_does_not_fall_back_to_older_answer_after_tool_planning(tmp_path):
    session = Session.create(make_config(tmp_path))
    with session.locked():
        session.append({"role": "assistant", "content": "old answer"})
        session.append(
            {
                "role": "assistant",
                "content": "planning text",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
            }
        )
    assert session.close().answer is None


def test_old_session_without_workdir_remains_loadable(tmp_path):
    config = make_config(tmp_path)
    session = Session.create(config)
    state_path = session.path / "session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("workdir", None)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    reopened = Session.open(config, session.state.session_id)
    assert reopened.state.workdir is None


def test_old_session_without_parent_session_id_is_a_root(tmp_path):
    config = make_config(tmp_path)
    session = Session.create(config)
    state_path = session.path / "session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("parent_session_id", None)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    reopened = Session.open(config, session.state.session_id)
    assert reopened.state.parent_session_id is None


def test_subagent_name_is_persisted_and_root_name_is_optional(tmp_path):
    config = make_config(tmp_path)
    parent = Session.create(config)
    child = Session.create(
        config, parent_session_id=parent.state.session_id, name="影像元数据检查员"
    )

    reopened = Session.open(config, child.state.session_id)
    assert reopened.state.name == "影像元数据检查员"
    assert reopened.state.parent_session_id == parent.state.session_id
    assert Session.open(config, parent.state.session_id).state.name is None
    assert next(state for state in Session.list(config) if state.session_id == child.state.session_id).name == "影像元数据检查员"


def test_old_session_without_name_remains_loadable_without_rewriting(tmp_path):
    config = make_config(tmp_path)
    session = Session.create(config, parent_session_id="old-parent")
    state_path = session.path / "session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("name")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    original = state_path.read_bytes()

    reopened = Session.open(config, session.state.session_id)

    assert reopened.state.name is None
    assert reopened.state.parent_session_id == "old-parent"
    assert state_path.read_bytes() == original


def test_execution_state_defaults_and_old_state_load(tmp_path):
    config = make_config(tmp_path)
    session = Session.create(config)
    assert session.state.execution_status == "idle"
    assert session.state.execution_id is None
    state_path = session.path / "session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for key in (
        "execution_status",
        "execution_id",
        "execution_answer",
        "execution_error",
        "execution_started_at",
        "execution_finished_at",
        "execution_owner_pid",
    ):
        state.pop(key, None)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    reopened = Session.open(config, session.state.session_id)
    assert reopened.state.execution_status == "idle"
    assert reopened.state.execution_id is None


def test_execution_claim_is_atomic_and_cancel_transitions(tmp_path):
    config = make_config(tmp_path)
    session = Session.create(config)
    session.update_execution("queued", execution_id="exec-1")
    first = Session.open(config, session.state.session_id)
    second = Session.open(config, session.state.session_id)
    assert first.claim_execution("exec-1") is True
    assert second.claim_execution("exec-1") is False
    assert Session.open(config, session.state.session_id).state.execution_status == "running"
    assert first.request_execution_cancel("exec-1") == "cancel_requested"
    assert Session.open(config, session.state.session_id).state.execution_status == "cancel_requested"
    assert first.update_execution("cancelled", execution_id="exec-1") == "cancelled"
    assert Session.open(config, session.state.session_id).state.execution_finished_at is not None


def test_queued_execution_cancel_is_immediate(tmp_path):
    config = make_config(tmp_path)
    session = Session.create(config)
    session.update_execution("queued", execution_id="exec-queued")
    assert session.request_execution_cancel("exec-queued") == "cancelled"
    reopened = Session.open(config, session.state.session_id)
    assert reopened.state.execution_status == "cancelled"
    assert reopened.state.execution_finished_at is not None


def test_completed_child_can_queue_a_new_execution_id(tmp_path):
    config = make_config(tmp_path)
    session = Session.create(config)
    session.update_execution("queued", execution_id="first")
    session.claim_execution("first")
    session.update_execution("succeeded", execution_id="first", answer="done")
    session.update_execution("queued", execution_id="second")
    reopened = Session.open(config, session.state.session_id)
    assert reopened.state.execution_id == "second"
    assert reopened.state.execution_status == "queued"
    assert reopened.state.execution_answer is None
    assert reopened.claim_execution("second") is True


def test_public_close_closes_only_direct_children(tmp_path):
    config = make_config(tmp_path)
    parent = Session.create(config)
    children = [
        Session.create(config, parent_session_id=parent.state.session_id)
        for _ in range(2)
    ]
    unrelated_parent = Session.create(config)
    unrelated_child = Session.create(config, parent_session_id=unrelated_parent.state.session_id)
    grandchild = Session.create(config, parent_session_id=children[0].state.session_id)

    result = close_session(config, parent.state.session_id)

    assert result.status == "closed"
    for session in [parent, *children]:
        assert Session.open(config, session.state.session_id).state.status == "closed"
    for session in [unrelated_parent, unrelated_child, grandchild]:
        assert Session.open(config, session.state.session_id).state.status == "running"


@pytest.mark.parametrize("parent_status", ["closed", "failed"])
def test_public_close_propagates_parent_status_and_preserves_terminal_children(tmp_path, parent_status):
    config = make_config(tmp_path)
    parent = Session.create(config)
    active = Session.create(config, parent_session_id=parent.state.session_id)
    terminal_children = [
        Session.create(config, parent_session_id=parent.state.session_id)
        for _ in range(2)
    ]
    terminal_children[0].close(status="closed", answer="completed", stop_reason="user_closed")
    terminal_children[1].close(status="failed", answer="partial", stop_reason="old_failure")
    original_states = [(child.path / "session.json").read_bytes() for child in terminal_children]

    result = close_session(config, parent.state.session_id, status=parent_status)

    assert result.status == parent_status
    assert Session.open(config, active.state.session_id).state.status == parent_status
    assert [(child.path / "session.json").read_bytes() for child in terminal_children] == original_states


def test_public_close_finishes_other_children_and_retries_after_parent_closed(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    parent = Session.create(config)
    children = [
        Session.create(config, parent_session_id=parent.state.session_id)
        for _ in range(3)
    ]
    failed_ids = {child.state.session_id for child in children[:2]}
    original_close = Session.close

    def interrupted_close(self, *args, **kwargs):
        if self.state.session_id in failed_ids:
            raise OSError("temporary child close failure")
        return original_close(self, *args, **kwargs)

    monkeypatch.setattr(Session, "close", interrupted_close)

    with pytest.raises(ConfigError) as caught:
        close_session(config, parent.state.session_id)

    assert caught.value.code == "subagent_close_failed"
    assert all(session_id in caught.value.message for session_id in failed_ids)
    assert "temporary child close failure" in caught.value.message
    assert Session.open(config, parent.state.session_id).state.status == "closed"
    assert Session.open(config, children[2].state.session_id).state.status == "closed"
    assert all(Session.open(config, session_id).state.status == "running" for session_id in failed_ids)

    failed_ids.clear()
    assert close_session(config, parent.state.session_id).status == "closed"
    assert all(Session.open(config, child.state.session_id).state.status == "closed" for child in children)


def test_public_close_waits_for_delegation_then_closes_parent_and_child(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    sub_prompt = tmp_path / "subagent.md"
    sub_prompt.write_text("Subagent working in {workdir}", encoding="utf-8")
    config = config.model_copy(update={
        "runtime": config.runtime.model_copy(update={"subagent_system_prompt_file": sub_prompt}),
    })
    parent = Session.create(config, workdir=tmp_path)
    child_started = Event()
    release_child = Event()
    close_started = Event()
    close_done = Event()
    outcomes = {}

    class BlockingClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            calls = []
            content = "parent done"
            if self.calls == 1:
                content = None
                calls = [SimpleNamespace(
                    id="delegate", type="function",
                    function=SimpleNamespace(
                        name="task", arguments=json.dumps({"name": "检查员", "prompt": "inspect"}),
                    ),
                )]
            elif self.calls == 2:
                child_started.set()
                assert release_child.wait(5), "child was not released"
                content = "child done"
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=content, tool_calls=calls),
            )])

    client = BlockingClient()
    original_close = Session.close

    def observed_close(self, *args, **kwargs):
        if self.state.session_id == parent.state.session_id:
            close_started.set()
        return original_close(self, *args, **kwargs)

    monkeypatch.setattr(Session, "close", observed_close)

    def interact():
        try:
            outcomes["run"] = run(
                parent.state.session_id, InputRequest(query="delegate"), config,
                openai_sdk_client=client,
            )
        except Exception as exc:
            outcomes["run"] = exc

    def close():
        try:
            outcomes["close"] = close_session(config, parent.state.session_id)
        except Exception as exc:
            outcomes["close"] = exc
        finally:
            close_done.set()

    interaction = Thread(target=interact, daemon=True)
    closer = Thread(target=close, daemon=True)
    interaction.start()
    try:
        assert child_started.wait(5), "delegation did not start"
        closer.start()
        assert close_started.wait(5), "close did not start"
        assert not close_done.wait(0.1), "close must wait for the active parent interaction"
    finally:
        release_child.set()
        interaction.join(timeout=5)
        if closer.ident is not None:
            closer.join(timeout=5)

    assert not interaction.is_alive() and not closer.is_alive(), "close and delegation deadlocked"
    assert not isinstance(outcomes["run"], Exception), outcomes["run"]
    assert not isinstance(outcomes["close"], Exception), outcomes["close"]
    assert outcomes["run"].status == "succeeded"
    assert outcomes["close"].status == "closed"
    states = Session.list(config)
    assert len(states) == 2
    assert all(state.status == "closed" for state in states)
    child = next(state for state in states if state.parent_session_id == parent.state.session_id)
    assert child.name == "检查员"


def test_create_accepts_custom_system_prompt_and_parent_link(tmp_path):
    config = make_config(tmp_path)
    parent = Session.create(config)
    # The override is the final rendered text; placeholder rendering is the
    # caller's job (agent_loop passes config.subagent_system_prompt(workdir)).
    child = Session.create(
        config,
        query="subtask",
        system_prompt="child prompt",
        parent_session_id=parent.state.session_id,
    )
    assert child.messages()[0] == {"role": "system", "content": "child prompt"}
    assert child.state.parent_session_id == parent.state.session_id
    # Without an override the configured prompt stays in charge.
    assert Session.create(config).messages()[0]["content"].startswith("system")


def test_messages_returns_fresh_list_shell(tmp_path):
    session = Session.create(make_config(tmp_path))
    with session.locked():
        session.append({"role": "user", "content": "hello"})
    messages = session.messages()
    messages.append({"role": "user", "content": "mutated by caller"})
    reopened = Session.open(make_config(tmp_path), session.state.session_id)
    assert len(reopened.messages()) == 2
    assert reopened.messages()[-1]["content"] == "hello"


def test_ledger_skips_corrupt_lines_and_continues_seq(tmp_path):
    config = make_config(tmp_path)
    session = Session.create(config)
    with session.locked():
        session.append({"role": "user", "content": "good"})
    with (session.path / "messages.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{corrupt json\n")
        handle.write('{"no_seq": true}\n')

    reopened = Session.open(config, session.state.session_id)
    assert [message["content"] for message in reopened.messages()][-1] == "good"
    with reopened.locked():
        reopened.append({"role": "user", "content": "after corruption"})

    final = Session.open(config, session.state.session_id)
    assert [message["content"] for message in final.messages()][-1] == "after corruption"
    ledger_path = session.path / "messages.jsonl"
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if '"seq"' in line
    ]
    assert [record["seq"] for record in records] == [1, 2, 3]


def test_list_returns_all_session_states_and_skips_strays(tmp_path):
    config = make_config(tmp_path)
    first = Session.create(config)
    second = Session.create(config)
    with second.locked():
        second.append({"role": "user", "content": "hi"})
    second.close()
    (config.runtime.sessions_dir / "stray.txt").write_text("not a session", encoding="utf-8")
    broken = config.runtime.sessions_dir / "broken"
    broken.mkdir()
    (broken / "session.json").write_text("{not json", encoding="utf-8")

    states = Session.list(config)

    assert [state.session_id for state in states] == sorted(
        [first.state.session_id, second.state.session_id]
    )
    by_id = {state.session_id: state for state in states}
    assert by_id[first.state.session_id].status == "running"
    assert by_id[second.state.session_id].status == "closed"
    assert by_id[second.state.session_id].message_count == 2


def test_list_with_missing_sessions_dir_returns_empty(tmp_path):
    assert Session.list(make_config(tmp_path)) == []
