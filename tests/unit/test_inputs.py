import pytest

from rs_harness.inputs import build_human_message


def test_build_human_message_rejects_empty_input():
    with pytest.raises(ValueError, match="empty_input"):
        build_human_message("  ")


def test_build_human_message_preserves_text():
    assert build_human_message(" hello ").content == " hello "
