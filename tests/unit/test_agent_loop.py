import json
from pathlib import Path
from types import SimpleNamespace

from rs_harness.agent_loop import InputRequest, create_session, run
from rs_harness.config import Config
from rs_harness.runtime import Session


def make_config(tmp_path: Path, *, max_turns: int = 20) -> Config:
    prompt = tmp_path / "system.md"
    prompt.write_text(
        "system\nWorking directory: {workdir}\n"
        "Only create, modify, or edit files inside this directory.",
        encoding="utf-8",
    )
    return Config(
        model={"model": "test-model", "api_key_env": "UNUSED_RS_KEY"},
        loop={"max_turns": max_turns},
        runtime={
            "sessions_dir": tmp_path / "sessions",
            "system_prompt_file": prompt,
            "dotenv_file": tmp_path / "missing.env",
        },
    )


class FakeOpenAI:
    def __init__(self, content, tool_calls=None):
        self.chat = self
        self.completions = self
        self.content = content
        self.tool_calls = tool_calls or []
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content, tool_calls=self.tool_calls))])


def test_empty_request_is_rejected_without_message(tmp_path):
    config = make_config(tmp_path)
    session = create_session(config)
    result = run(session.state.session_id, InputRequest(), config)
    assert result.status == "failed"
    assert result.errors[0].code == "empty_input"
    assert session.messages() == [
        {
            "role": "system",
            "content": (
                f"system\nWorking directory: {session.state.workdir}\n"
                "Only create, modify, or edit files inside this directory."
            ),
        }
    ]


def test_image_only_does_not_initialize_model(tmp_path):
    config = make_config(tmp_path)
    session = create_session(config)
    result = run(session.state.session_id, InputRequest(image="scene.tif"), config)
    assert result.status == "not_supported"
    assert result.session_status == "running"


def test_openai_answer_is_recorded_and_context_reused(tmp_path):
    config = make_config(tmp_path)
    session = create_session(config)
    fake = FakeOpenAI("answer")
    result = run(session.state.session_id, InputRequest(query="question"), config, openai_sdk_client=fake)
    assert result.status == "succeeded"
    assert result.answer == "answer"
    assert fake.calls[0]["messages"][0] == session.messages()[0]
    assert len(json.loads((session.path / "messages.jsonl").read_text().splitlines()[-1])["message"]) >= 2
    fake.content = "follow-up"
    run(session.state.session_id, InputRequest(query="again"), config, openai_sdk_client=fake)
    assert fake.calls[1]["messages"][-2]["content"] == "answer"
    assert Session.open(config, session.state.session_id).state.query == "question"


def test_openai_tool_request_is_executed_and_budgeted(tmp_path):
    config = make_config(tmp_path, max_turns=1)
    session = create_session(config)
    fake = FakeOpenAI(None, [SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(name="x", arguments="{}"))])
    result = run(session.state.session_id, InputRequest(query="question"), config, openai_sdk_client=fake)
    assert result.status == "failed"
    assert result.errors[0].code == "budget_exceeded"
    assert fake.calls[0]["tools"]
    messages = session.messages()
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Unknown tool: x",
    }


def test_inline_image_data_is_not_forwarded_or_persisted(tmp_path):
    config = make_config(tmp_path)
    session = create_session(config)
    fake = FakeOpenAI("cannot inspect")
    inline = "data:image/tiff;base64," + "A" * 2048
    result = run(session.state.session_id, InputRequest(query="what is here?", image=inline), config, openai_sdk_client=fake)
    assert result.status == "succeeded"
    assert inline not in str(fake.calls[0]["messages"])
    assert inline not in (session.path / "messages.jsonl").read_text(encoding="utf-8")


def test_closed_session_reports_closed_status(tmp_path):
    config = make_config(tmp_path)
    session = create_session(config)
    session.close()
    result = run(session.state.session_id, InputRequest(query="late"), config, openai_sdk_client=FakeOpenAI("unused"))
    assert result.status == "failed"
    assert result.errors[0].code == "session_closed"
    assert result.session_status == "closed"


def test_legacy_session_without_system_context_is_rejected(tmp_path):
    config = make_config(tmp_path)
    session = create_session(config)
    state_path = session.path / "session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("system_context_version")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = run(
        session.state.session_id,
        InputRequest(query="question"),
        config,
        openai_sdk_client=FakeOpenAI("unused"),
    )

    assert result.status == "failed"
    assert result.errors[0].code == "session_context_missing"
