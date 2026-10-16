import sys
from io import StringIO

import pytest
from rich.cells import cell_len
from rich.console import Console

from rs_harness.mcp_display import print_mcp_servers


@pytest.fixture
def servers():
    return [
        {"name": "docs", "transport": "http", "tools": [
            {"raw_name": "search", "permission": "allow", "description": "文" * 130 + " description_end",
             "name": "mcp__docs__search", "input_schema": {"required": ["query"]}},
            {"raw_name": "长" * 80 + "tool_end", "permission": "confirm", "description": ""},
        ]},
        {"name": "imagery", "transport": "stdio", "tools": []},
    ]


@pytest.mark.parametrize("width", [32, 40, 60, 71, 72, 80, 100, 140])
def test_mcp_panel_preserves_full_catalog_and_fits_terminal(servers, width):
    stream = StringIO()
    print_mcp_servers(servers, console=Console(file=stream, width=width, force_terminal=False))
    output = stream.getvalue()
    normalized = "".join(output.replace("│", "").split())
    assert output.count("╭") == output.count("╰") == 1
    assert "2 servers · 2 tools" in output
    assert "docs · http · 2 tools" in output and "imagery · stdio · 0 tools" in output
    assert output.index("docs") < output.index("imagery")
    assert output.count("文") == 130 and "description_end" in normalized
    assert output.count("长") == 80 and "tool_end" in normalized
    assert "allow" in output and "confirm" in output and "—" in output
    assert "No tools exposed." in output
    assert "mcp__" not in output and "required" not in output and "query" not in output
    assert "\x1b" not in output and "..." not in output and "…" not in output
    assert all(cell_len(line) <= min(width, 100) for line in output.splitlines())
    if width < 72:
        assert "search [allow]" in output and "Description" not in output
    else:
        assert "Policy" in output and "Description" in output


@pytest.mark.parametrize("width", [32, 80])
def test_mcp_empty_config_uses_panel(width):
    stream = StringIO()
    print_mcp_servers([], console=Console(file=stream, width=width, force_terminal=False))
    output = stream.getvalue()
    assert "0 servers · 0 tools" in output and "No MCP servers configured." in output
    assert output.count("╭") == output.count("╰") == 1


@pytest.mark.parametrize("field", ["name", "transport", "raw_name", "description"])
@pytest.mark.parametrize("width", [40, 80])
def test_mcp_remote_text_is_literal_and_control_characters_are_escaped(field, width):
    unsafe = "[bold]bad[/bold]\x1b]52;c;dGVzdA==\x07\x9d52;c;dGVzdA==\x9c"
    tool = {"raw_name": "search", "permission": "allow", "description": ""}
    server = {"name": "docs", "transport": "http", "tools": [tool]}
    (server if field in ("name", "transport") else tool)[field] = unsafe
    stream = StringIO()
    print_mcp_servers([server], console=Console(file=stream, width=width, force_terminal=False))
    output = stream.getvalue()
    normalized = "".join(output.replace("│", "").split())
    assert "[bold]bad[/bold]" in normalized
    for escaped in (r"\x1b", r"\x07", r"\x9d", r"\x9c"):
        assert escaped in normalized
    assert not any(char in output for char in ("\x1b", "\x07", "\x9b", "\x9c", "\x9d"))


@pytest.mark.parametrize("tty,no_color", [(False, False), (True, True), (True, False)])
def test_mcp_default_console_respects_tty_and_no_color(servers, monkeypatch, tty, no_color):
    stream = StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: tty)
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "80")
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    print_mcp_servers(servers)
    output = stream.getvalue()
    if tty and not no_color:
        assert "\x1b[32mallow" in output and "\x1b[33mconfirm" in output
    else:
        assert "\x1b" not in output
