from pathlib import Path
from types import SimpleNamespace

from rs_harness.agent_loop import InputRequest, run
from rs_harness.config import Config


def make_config(tmp_path: Path, provider: str = "anthropic") -> Config:
    prompt = tmp_path / "system.md"
    prompt.write_text("system", encoding="utf-8")
    return Config(
        model={
            "provider": provider,
            "model": "test-model",
            "api_key_env": "UNUSED_RS_KEY",
        },
        runtime={
            "sessions_dir": tmp_path / "sessions",
            "system_prompt_file": prompt,
            "dotenv_file": tmp_path / "missing.env",
        },
    )


class FakeAnthropic:
    def __init__(self, content):
        self.messages = self
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="msg_1", model="test-model", stop_reason="end_turn", content=self.content
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
        return SimpleNamespace(
            id="chatcmpl_1",
            model="test-model",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=self.content, tool_calls=self.tool_calls),
                )
            ],
        )


def test_empty_request_is_silent(tmp_path):
    result = run(InputRequest(), make_config(tmp_path))
    assert result is None
    assert not (tmp_path / "sessions").exists()


def test_image_only_does_not_initialize_model(tmp_path):
    result = run(InputRequest(image="scene.tif"), make_config(tmp_path))
    assert result is not None
    assert result.status == "not_supported"
    assert result.stop_reason == "image_only_not_implemented"


def test_anthropic_answer_is_recorded(tmp_path):
    fake = FakeAnthropic([SimpleNamespace(type="text", text="answer")])
    result = run(InputRequest(query="question"), make_config(tmp_path), anthropic_sdk_client=fake)
    assert result is not None and result.status == "succeeded"
    assert result.answer == "answer"
    assert fake.calls[0]["tools"] == []
    assert (Path(result.session_path) / "events.jsonl").exists()


def test_openai_tool_request_is_explicitly_unsupported(tmp_path):
    fake = FakeOpenAI(
        None,
        [SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(name="x", arguments="{}"))],
    )
    result = run(
        InputRequest(query="question"), make_config(tmp_path, "openai"), openai_sdk_client=fake
    )
    assert result is not None and result.status == "failed"
    assert result.errors[0].code == "unsupported_action"
    assert fake.calls[0]["tools"] == []


def test_inline_image_data_is_not_forwarded_or_persisted(tmp_path):
    fake = FakeAnthropic([SimpleNamespace(type="text", text="cannot inspect")])
    inline = "data:image/tiff;base64," + "A" * 2048
    result = run(
        InputRequest(query="what is here?", image=inline),
        make_config(tmp_path),
        anthropic_sdk_client=fake,
    )
    assert result is not None and result.status == "succeeded"
    sent = fake.calls[0]["messages"][0]["content"]
    assert inline not in sent
    session = (Path(result.session_path) / "session.json").read_text(encoding="utf-8")
    assert inline not in session
