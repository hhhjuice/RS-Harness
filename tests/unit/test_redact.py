import re

from rs_harness.runtime.redact import (
    CREDENTIAL_KEYS,
    identity,
    redact_text,
    redact_value,
    secret_redactor,
    summarize_data_uri,
)


def test_redact_text_replaces_configured_secret():
    assert redact_text("key sk-live-abc tail", secret_redactor("sk-live-abc")) == "key [REDACTED] tail"


def test_redact_text_replaces_inline_data_uri_anywhere():
    text = "prefix data:image/png;base64,AAA suffix"
    assert redact_text(text) == "prefix [DATA_URI_REDACTED] suffix"


def test_redact_text_rewrites_credential_pairs_even_with_identity_redactor():
    assert redact_text("token=secret-value", identity) == "token=[REDACTED]"
    assert redact_text('api_key: "hunter2"') == "api_key: [REDACTED]"
    assert redact_text("Authorization: Bearer abc.def") == "Authorization: [REDACTED]"


def test_redact_text_leaves_ordinary_text_alone():
    assert redact_text("ordinary prose with no secrets") == "ordinary prose with no secrets"


def test_redact_value_walks_structures_and_exceptions():
    value = {"token": "keep", "nested": ["password=hunter2"], "err": ValueError("boom token=xyz")}
    scrubbed = redact_value(value)
    assert scrubbed == {
        "token": "keep",
        "nested": ["password=[REDACTED]"],
        "err": {"type": "ValueError", "message": "boom token=[REDACTED]"},
    }
    assert isinstance(scrubbed["err"]["message"], str)


def test_secret_redactor_empty_secret_is_identity():
    assert secret_redactor("") is identity
    assert secret_redactor("")("sk-anything") == "sk-anything"


def test_summarize_data_uri_only_matches_whole_values():
    summary = summarize_data_uri("data:image/tiff;base64," + "A" * 50)
    assert summary is not None
    assert summary.startswith("[data value omitted; length=73 sha256=")
    assert summarize_data_uri("data: notes from the field") is None


def test_credential_keys_cover_known_secret_field_names():
    assert {"api_key", "authorization", "token", "password"} <= CREDENTIAL_KEYS
    assert all(re.fullmatch(r"[a-z_-]+", key) for key in CREDENTIAL_KEYS)
