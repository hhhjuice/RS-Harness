import base64
from pathlib import Path
from types import SimpleNamespace

from rs_harness.agent_loop import InputRequest, create_session, run
from rs_harness.config import Config

JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-payload"
TIFF = b"II*\x00" + b"tiff-payload"


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def _config(tmp_path: Path) -> Config:
    prompt = tmp_path / "system.md"
    prompt.write_text("system", encoding="utf-8")
    return Config(
        model={"model": "test-model", "api_key_env": "UNUSED_RS_KEY"},
        loop={"max_turns": 20},
        runtime={
            "sessions_dir": tmp_path / "sessions",
            "system_prompt_file": prompt,
            "dotenv_file": tmp_path / "missing.env",
        },
    )


def _response(content=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=[], role="assistant")
            )
        ]
    )


def _user_contents(call):
    return [
        message["content"]
        for message in call["messages"]
        if message.get("role") == "user"
    ]


def test_local_image_is_sent_as_vision_block_and_ledger_keeps_reference(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    jpg = workspace / "photo.jpg"
    jpg.write_bytes(JPEG)
    client = SequenceClient([_response("a photo")])
    session = create_session(config, workdir=workspace)

    result = run(
        session.state.session_id,
        InputRequest(query="描述这张图片", image=str(jpg)),
        config,
        openai_sdk_client=client,
    )

    assert result.status == "succeeded"
    content = _user_contents(client.calls[0])[-1]
    assert isinstance(content, list)
    assert content[0] == {
        "type": "text",
        "text": f"描述这张图片\nImage attached: {jpg.resolve()}",
    }
    expected_url = "data:image/jpeg;base64," + base64.b64encode(JPEG).decode("ascii")
    assert content[1] == {"type": "image_url", "image_url": {"url": expected_url}}

    ledger = (session.path / "messages.jsonl").read_text(encoding="utf-8")
    assert "image_ref" in ledger
    assert "jpeg-payload" not in ledger
    assert "data:image/jpeg" not in ledger


def test_image_path_typed_into_query_text_is_auto_attached(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    jpg = workspace / "photo.jpg"
    jpg.write_bytes(JPEG)
    client = SequenceClient([_response("seen")])
    session = create_session(config, workdir=workspace)

    result = run(
        session.state.session_id,
        {"query": f"描述这张图像。{jpg}"},
        config,
        openai_sdk_client=client,
    )

    assert result.status == "succeeded"
    content = _user_contents(client.calls[0])[-1]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_history_reembeds_image_and_degrades_when_file_removed(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    jpg = workspace / "photo.jpg"
    jpg.write_bytes(JPEG)
    client = SequenceClient([_response("one"), _response("two"), _response("three")])
    session = create_session(config, workdir=workspace)
    session_id = session.state.session_id

    run(session_id, {"query": "first", "image": str(jpg)}, config, openai_sdk_client=client)
    run(session_id, {"query": "second"}, config, openai_sdk_client=client)

    history_content = _user_contents(client.calls[1])[0]
    assert history_content[1]["type"] == "image_url"
    assert history_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    jpg.unlink()
    run(session_id, {"query": "third"}, config, openai_sdk_client=client)

    degraded = _user_contents(client.calls[2])[0]
    assert degraded[1] == {
        "type": "text",
        "text": f"Image previously attached at {jpg.resolve()} is no longer available.",
    }


def test_remote_sensing_format_degrades_to_opaque_reference(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tif = workspace / "scene.tif"
    tif.write_bytes(TIFF)
    client = SequenceClient([_response("cannot view")])
    session = create_session(config, workdir=workspace)

    result = run(
        session.state.session_id,
        InputRequest(query="分析这张影像", image=str(tif)),
        config,
        openai_sdk_client=client,
    )

    assert result.status == "succeeded"
    content = _user_contents(client.calls[0])[-1]
    assert isinstance(content, str)
    assert "not directly viewable" in content
    assert "MCP" in content
    assert "image_url" not in str(client.calls[0]["messages"])


def test_http_image_url_is_passed_through(tmp_path: Path):
    config = _config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SequenceClient([_response("web image")])
    session = create_session(config, workdir=workspace)

    result = run(
        session.state.session_id,
        InputRequest(query="这是什么", image="https://example.com/a.jpg"),
        config,
        openai_sdk_client=client,
    )

    assert result.status == "succeeded"
    content = _user_contents(client.calls[0])[-1]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.jpg"},
    }
    ledger = (session.path / "messages.jsonl").read_text(encoding="utf-8")
    assert "https://example.com/a.jpg" in ledger
    assert "data:" not in ledger
