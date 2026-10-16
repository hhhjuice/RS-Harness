import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console

from rs_harness.skill_display import print_skill_catalog
from rs_harness.skills import Skill


@pytest.fixture
def skills():
    return [
        Skill("search", "文" * 130 + " description_end", Path("/private/search/SKILL.md"), "PRIVATE-BODY"),
        Skill("长" * 80 + "name_end", "Second description", Path("/private/second/SKILL.md"), "PRIVATE-BODY"),
    ]


@pytest.mark.parametrize("width", [32, 40, 60, 71, 72, 80, 100, 140])
def test_skill_panel_preserves_full_metadata_and_fits_terminal(skills, width):
    directory = Path("/project") / ("目录" * 65) / "skill library"
    stream = StringIO()
    print_skill_catalog(directory, skills, console=Console(file=stream, width=width, force_terminal=False))
    output = stream.getvalue()
    normalized = "".join(output.replace("│", "").split())
    assert output.count("╭") == output.count("╰") == 1
    assert "Installed skills" in output and "2 skills" in output
    assert "".join(f"Directory: {directory}".split()) in normalized
    assert output.index("search") < output.index("长")
    assert output.count("文") == 130 and "description_end" in normalized
    assert output.count("长") == 80 and "name_end" in normalized
    assert "Seconddescription" in normalized
    assert "PRIVATE-BODY" not in output and "/private" not in output and "SKILL.md" not in output
    assert "\x1b" not in output and "..." not in output and "…" not in output
    assert all(cell_len(line) <= min(width, 100) for line in output.splitlines())
    assert ("Description" in output) == (width >= 72)


@pytest.mark.parametrize("width", [32, 80])
def test_skill_empty_catalog_keeps_directory_in_panel(width):
    stream = StringIO()
    print_skill_catalog(Path("/project/skills"), [], console=Console(file=stream, width=width, force_terminal=False))
    output = stream.getvalue()
    assert "0 skills" in output and "No skills installed." in output
    assert "Directory: /project/skills" in output
    assert output.count("╭") == output.count("╰") == 1


@pytest.mark.parametrize("field", ["directory", "name", "description"])
@pytest.mark.parametrize("width", [40, 80])
def test_skill_metadata_is_literal_and_control_characters_are_escaped(field, width):
    unsafe = "[bold]bad[/bold]\x1b]52;c;dGVzdA==\x07\x9d52;c;dGVzdA==\x9c"
    directory = Path("/project") / unsafe if field == "directory" else Path("/project/skills")
    skill = Skill(unsafe if field == "name" else "example", unsafe if field == "description" else "",
                  Path("/private/SKILL.md"), "PRIVATE-BODY")
    stream = StringIO()
    print_skill_catalog(directory, [skill], console=Console(file=stream, width=width, force_terminal=False))
    output = stream.getvalue()
    normalized = "".join(output.replace("│", "").split())
    assert "[bold]bad[/bold]" in normalized
    for escaped in (r"\x1b", r"\x07", r"\x9d", r"\x9c"):
        assert escaped in normalized
    assert not any(char in output for char in ("\x1b", "\x07", "\x9b", "\x9c", "\x9d"))


@pytest.mark.parametrize("tty,no_color", [(False, False), (True, True), (True, False)])
def test_skill_default_console_respects_tty_and_no_color(skills, monkeypatch, tty, no_color):
    stream = StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: tty)
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "80")
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    print_skill_catalog(Path("/project/skills"), skills)
    output = stream.getvalue()
    if tty and not no_color:
        assert "\x1b[1;36msearch" in output
    else:
        assert "\x1b" not in output
