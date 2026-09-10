from rs_harness.runtime.tools import dispatch_tool, safe_path


def test_safe_path_escape_requires_explicit_flag(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    try:
        safe_path(outside, tmp_path)
    except ValueError as exc:
        assert "escapes workspace" in str(exc)
    else:
        raise AssertionError("workspace escape should be rejected by default")

    assert safe_path(outside, tmp_path, allow_workspace_escape=True) == outside.resolve()


def test_dispatch_file_tool_allows_only_explicit_workspace_escape(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    denied = dispatch_tool("read_file", {"path": str(outside)}, tmp_path)
    assert "escapes workspace" in denied

    allowed = dispatch_tool(
        "read_file",
        {"path": str(outside)},
        tmp_path,
        allow_workspace_escape=True,
    )
    assert allowed == "secret"


def test_dispatch_write_escape_is_scoped_to_single_call(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "created.txt"
    args = {"path": str(outside), "content": "ok"}

    denied = dispatch_tool("write_file", args, tmp_path)
    assert "escapes workspace" in denied
    assert not outside.exists()

    allowed = dispatch_tool("write_file", args, tmp_path, allow_workspace_escape=True)
    assert "Wrote" in allowed
    assert outside.read_text(encoding="utf-8") == "ok"
