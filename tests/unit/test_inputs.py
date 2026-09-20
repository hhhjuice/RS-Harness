import base64
import hashlib

import pytest
from langchain_core.messages import HumanMessage

from rs_harness.inputs import (
    InputError,
    build_human_message,
    summarize_message_content,
    validate_human_message,
    validate_image_messages,
)
from tests.images import write_test_image


def test_build_human_message_rejects_empty_input():
    with pytest.raises(ValueError, match="empty_input"):
        build_human_message("  ")


def test_build_human_message_preserves_text():
    assert build_human_message(" hello ").content == " hello "


def test_standard_blocks_local_url_order_and_image_only(tmp_path):
    data = write_test_image(tmp_path / "sample.bin")
    sources = ["sample.bin", "https://example.com/image-without-extension"]
    message = build_human_message(" describe ", images=sources, workdir=tmp_path)
    assert message.content == [
        {"type": "text", "text": " describe "},
        {"type": "image", "base64": base64.b64encode(data).decode(), "mime_type": "image/png"},
        {"type": "image", "url": sources[1]},
    ]
    assert build_human_message(images=[str(tmp_path / "sample.bin")], workdir=tmp_path / "other").content == [message.content[1]]
    validate_human_message(build_human_message(images=sources, workdir=tmp_path))


@pytest.mark.parametrize("source", ["missing.png", ".", "file:///tmp/image.png", "data:image/png;base64,aGVsbG8=", "ftp://example.com/image", "https:///image", "https://[invalid", ""])
def test_invalid_sources_fail_before_submission(tmp_path, source):
    with pytest.raises(InputError):
        build_human_message(images=[source], workdir=tmp_path)


@pytest.mark.parametrize("content", [b"", b"not an image", b"BMwrong-format"])
def test_empty_and_unsupported_files(tmp_path, content):
    (tmp_path / "fake.png").write_bytes(content)
    with pytest.raises(InputError):
        build_human_message(images=["fake.png"], workdir=tmp_path)


@pytest.mark.parametrize("header,mime", [
    (b"\xff\xd8\xff", "image/jpeg"), (b"GIF89a", "image/gif"), (b"RIFF1234WEBP", "image/webp"),
])
def test_mime_comes_from_content(tmp_path, header, mime):
    (tmp_path / "image.unknown").write_bytes(header)
    message = build_human_message(images=["image.unknown"], workdir=tmp_path)
    assert message.content[0]["mime_type"] == mime


def test_deepseek_limits_are_applied_only_to_the_documented_provider(config, tmp_path):
    deepseek = config.model_copy(update={"model": config.model.model_copy(update={"base_url": "https://api.deepseek.com", "model": "deepseek-flash"})})
    oversized = tmp_path / "large.png"
    with oversized.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(InputError, match="image_too_large"):
        build_human_message(images=[str(oversized)], config=deepseek)
    with pytest.raises(InputError, match="too_many_images"):
        build_human_message(images=["https://example.com/image"] * 601, config=deepseek)
    with pytest.raises(InputError, match="image_url_too_long"):
        build_human_message(images=["https://example.com/" + "a" * 8192], config=deepseek)
    messages = [HumanMessage(content_blocks=[{"type": "image", "url": "https://example.com/image"}])] * 601
    with pytest.raises(InputError, match="too_many_images"):
        validate_image_messages(messages, deepseek)
    validate_image_messages(messages, config)


def test_local_image_read_is_capped_for_every_provider(config, tmp_path):
    oversized = tmp_path / "large.png"
    with oversized.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(InputError, match="image_too_large: local image files are limited to 32 MiB"):
        build_human_message(images=[str(oversized)], config=config)
    with pytest.raises(InputError, match="image_too_large: local image files are limited to 32 MiB"):
        build_human_message(images=[str(oversized)])
    undersized = tmp_path / "fit.png"
    with undersized.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.truncate(32 * 1024 * 1024)
    assert build_human_message(images=[str(undersized)], config=config).content[0]["base64"]


def test_summarize_message_content_replaces_inline_images_with_digests(tmp_path):
    data = write_test_image(tmp_path / "sample.bin")
    message = build_human_message(
        " describe ", images=["sample.bin", "https://example.com/image"], workdir=tmp_path,
    )
    assert summarize_message_content(message) == [
        {"type": "text", "text": " describe "},
        {"type": "image", "mime_type": "image/png", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
        {"type": "image", "url": "https://example.com/image"},
    ]
    assert "base64" not in str(summarize_message_content(message))
    assert summarize_message_content(HumanMessage(content=" hello ")) == " hello "


def test_unknown_profile_is_not_rejected_and_known_unsupported_model_is(config):
    message = build_human_message(images=["https://example.com/image"])
    validate_image_messages([message], config, {})
    with pytest.raises(InputError, match="image_inputs_not_supported"):
        validate_image_messages([message], config, {"image_inputs": False})
    custom = config.model_copy(update={"model": config.model.model_copy(update={"base_url": "https://example.com/v1"})})
    validate_image_messages([message], custom, {"image_inputs": False})


@pytest.mark.parametrize("content", [" ", [], [{"type": "text", "text": " "}]])
def test_direct_empty_messages_are_rejected(content):
    with pytest.raises(ValueError, match="empty_input"):
        validate_human_message(HumanMessage(content=content))
