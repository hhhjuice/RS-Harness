import json
from pathlib import Path

import pytest

from rs_harness.runtime import Session


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
