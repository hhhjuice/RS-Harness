"""Responsive terminal rendering for the installed skill directory."""

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .skills import Skill


def _terminal_text(value: str) -> str:
    """Display metadata and paths literally, including terminal control characters."""
    return "".join(char if char.isprintable() else char.encode("unicode_escape").decode("ascii") for char in value)


def print_skill_catalog(skills_dir: Path, skills: Sequence[Skill], *, console: Console | None = None) -> None:
    """Print names and full descriptions without exposing skill bodies or changing the catalog."""
    console = console or Console(highlight=False, force_terminal=sys.stdout.isatty(),
                                 color_system=None if "NO_COLOR" in os.environ else "auto")
    width = min(console.width, 100)
    items = [
        Text(f"{len(skills)} skill{'' if len(skills) == 1 else 's'}", style="dim"),
        Text(f"Directory: {_terminal_text(str(skills_dir))}", style="dim", overflow="fold"),
        Text(""),
    ]
    if not skills:
        items.append(Text("No skills installed.", style="dim"))
    elif width < 72:
        for index, skill in enumerate(skills):
            if index:
                items.append(Text(""))
            items.extend([
                Text(_terminal_text(skill.name), style="bold cyan", overflow="fold"),
                Padding(Text(_terminal_text(skill.description), overflow="fold"), (0, 0, 0, 2)),
            ])
    else:
        table = Table(box=None, padding=(0, 1), expand=True, show_edge=False, header_style="dim")
        table.add_column("Skill", ratio=3, overflow="fold")
        table.add_column("Description", ratio=5, overflow="fold")
        for skill in skills:
            table.add_row(Text(_terminal_text(skill.name), style="bold cyan"), Text(_terminal_text(skill.description)))
        items.append(table)

    console.print(Panel(Group(*items), title=Text("Installed skills", style="bold"), title_align="left",
                        box=box.ROUNDED, border_style="dim", width=width, padding=(0, 1)))
