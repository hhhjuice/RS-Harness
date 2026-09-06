import json
from pathlib import Path

import pytest

from rs_harness.runtime import Session


def make_config(tmp_path: Path):
    from rs_harness.config import Config

    prompt = tmp_path / "system.md"
    prompt.write_text("system", encoding="utf-8")
    return Config(
        model={"model": "test-model", "api_key_env": "UNUSED_RS_KEY"},
        runtime={
            "sessions_dir": tmp_path / "sessions",
            "system_prompt_file": prompt,
            "dotenv_file": tmp_path / "missing.env",
        },
    )


def test_create_open_and_append_complete_message_ledger(tmp_path):
    session = Session.create(make_config(tmp_path))
    assert session.state.status == "running"
    assert session.messages() == [{"role": "system", "content": "system"}]

    session.append_message({"role": "user", "content": "hello"})
    session.append_message(
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "c1"}]}
    )
    reopened = Session.open(make_config(tmp_path), session.state.session_id)
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
    with pytest.raises(ValueError, match="session_closed"):
        session.append_message({"role": "user", "content": "late"})


def test_ledger_redacts_credentials_and_data_values(tmp_path):
    session = Session.create(make_config(tmp_path))
    session.append_message(
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
    session.append_message({"role": "assistant", "content": "derived answer"})
    result = session.close()
    assert result.answer == "derived answer"
    assert session.state.answer == "derived answer"


def test_close_derives_text_blocks_and_explicit_answer_wins(tmp_path):
    session = Session.create(make_config(tmp_path))
    session.append_message(
        {"role": "assistant", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}
    )
    assert session.close().answer == "one\ntwo"

    explicit = Session.create(make_config(tmp_path))
    explicit.append_message({"role": "assistant", "content": "ledger answer"})
    assert explicit.close(answer="explicit answer").answer == "explicit answer"
