import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

from rs_harness.context import AgentContext
from rs_harness.inputs import MAX_IMAGE_BYTES, InputError, load_local_image
from rs_harness.tools.filesystem import read_image


@pytest.mark.parametrize("header,mime", [
    (b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"), (b"RIFF1234WEBP", "image/webp"),
])
@pytest.mark.parametrize("absolute", [False, True])
def test_read_image_uses_content_and_session_workdir(tmp_path, monkeypatch, header, mime, absolute):
    image = tmp_path / "wrong-extension.txt"
    image.write_bytes(header)
    runtime = SimpleNamespace(context=AgentContext(str(tmp_path), max_tool_output_bytes=1))
    monkeypatch.chdir(tmp_path.parent)
    result = read_image.func(str(image) if absolute else image.name, runtime)
    assert result == [
        {"type": "text", "text": "Image loaded from wrong-extension.txt"},
        {"type": "image", "mime_type": mime, "base64": base64.b64encode(header).decode("ascii")},
    ]


@pytest.mark.parametrize("path,data,reason", [
    ("missing.png", None, "image_not_found"), (".", None, "Not a regular file"),
    ("empty.png", b"", "empty_image"), ("text.png", b"text", "unsupported_image_format"),
    ("scene.tif", b"II\x2a\x00", "unsupported_image_format"),
    ("https://example.com/image.png", None, "image_not_found"),
])
def test_read_image_reports_file_errors(tmp_path, path, data, reason):
    if data is not None:
        (tmp_path / path).write_bytes(data)
    runtime = SimpleNamespace(context=AgentContext(str(tmp_path)))
    with pytest.raises(ToolException, match=reason):
        read_image.func(path, runtime)


def test_read_image_reports_os_read_failure(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    image.touch()

    def denied(*args, **kwargs):
        raise PermissionError("read denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(ToolException, match="read_image: unable_to_read_image.*read denied"):
        read_image.func(image.name, SimpleNamespace(context=AgentContext(str(tmp_path))))


@pytest.mark.parametrize("path", [None, 3, True, [], "", "   "])
def test_read_image_rejects_invalid_arguments(path):
    with pytest.raises(ValidationError):
        read_image.args_schema.model_validate({"path": path})
    assert set(read_image.tool_call_schema.model_fields) == {"path"}


@pytest.mark.parametrize("size", [MAX_IMAGE_BYTES - 1, MAX_IMAGE_BYTES, MAX_IMAGE_BYTES + 1])
def test_read_image_size_boundary(tmp_path, size):
    image = tmp_path / "image.png"
    with image.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.truncate(size)
    runtime = SimpleNamespace(context=AgentContext(str(tmp_path)))
    if size > MAX_IMAGE_BYTES:
        with pytest.raises(ToolException, match="image_too_large"):
            read_image.func(image.name, runtime)
    else:
        assert len(base64.b64decode(read_image.func(image.name, runtime)[1]["base64"])) == size


def test_local_image_read_is_bounded_and_budget_checked_before_encoding(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    image.touch()
    reads = []

    class ProbeFile(BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    def encode(data):
        pytest.fail("Image must be rejected before encoding")

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: ProbeFile(b"\x89PNG\r\n\x1a\n"))
    monkeypatch.setattr("rs_harness.inputs.base64.b64encode", encode)
    with pytest.raises(InputError, match="image_payload_too_large"):
        load_local_image(image, max_encoded_bytes=12)
    assert reads == [MAX_IMAGE_BYTES + 1]
