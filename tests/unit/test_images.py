import hashlib
from pathlib import Path

import rs_harness.runtime.images as images
from rs_harness.runtime.images import inspect_local_image, sniff_mime

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 12
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
GIF = b"GIF89a" + b"\x00" * 12
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 4
TIFF = b"II*\x00" + b"\x00" * 12


def test_sniff_mime_matches_supported_signatures():
    assert sniff_mime(JPEG) == "image/jpeg"
    assert sniff_mime(PNG) == "image/png"
    assert sniff_mime(GIF) == "image/gif"
    assert sniff_mime(WEBP) == "image/webp"
    assert sniff_mime(TIFF) is None
    assert sniff_mime(b"plain text file") is None
    assert sniff_mime(b"") is None


def test_inspect_local_image_builds_spec_and_reports_reasons(tmp_path: Path):
    jpg = tmp_path / "photo.jpg"
    jpg.write_bytes(JPEG)

    attachment, reason = inspect_local_image(jpg)

    assert reason == ""
    assert attachment is not None
    assert attachment.mime == "image/jpeg"
    assert attachment.bytes == len(JPEG)
    assert attachment.sha256 == hashlib.sha256(JPEG).hexdigest()
    assert attachment.path == str(jpg.resolve())

    tif = tmp_path / "scene.tif"
    tif.write_bytes(TIFF)
    attachment, reason = inspect_local_image(tif)
    assert attachment is None
    assert "unsupported format" in reason

    attachment, reason = inspect_local_image(tmp_path / "missing.jpg")
    assert attachment is None
    assert "not found" in reason


def test_inspect_local_image_rejects_oversized_files(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 8)
    big = tmp_path / "big.jpg"
    big.write_bytes(JPEG + b"x" * 16)

    attachment, reason = inspect_local_image(big)

    assert attachment is None
    assert "too large" in reason
