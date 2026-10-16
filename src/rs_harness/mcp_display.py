"""Responsive terminal rendering for the configured MCP tool directory."""

import os
import sys
from typing import Any, Sequence

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


def _terminal_text(value: str) -> str:
    """Render remote control characters literally instead of executing them."""
    return "".join(char if char.isprintable() else char.encode("unicode_escape").decode("ascii") for char in value)


def _count_label(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def print_mcp_servers(servers: Sequence[dict[str, Any]], *, console: Console | None = None) -> None:
    """Print all discovered servers and tools in one panel, without truncating text."""
    console = console or Console(highlight=False, force_terminal=sys.stdout.isatty(),
                                 color_system=None if "NO_COLOR" in os.environ else "auto")
    width = min(console.width, 100)
    narrow = width < 72
    tool_count = sum(len(server["tools"]) for server in servers)
    items = [Text(f"{_count_label(len(servers), 'server')} · {_count_label(tool_count, 'tool')}", style="dim"), Text("")]
    if not servers:
        items.append(Text("No MCP servers configured.", style="dim"))

    for index, server in enumerate(servers):
        if index:
            items.append(Rule(style="dim"))
        tools = server["tools"]
        header = Text(overflow="fold")
        header.append(_terminal_text(server["name"]), style="bold cyan")
        header.append(f" · {_terminal_text(server['transport'])} · {_count_label(len(tools), 'tool')}", style="dim")
        items.append(header)
        if not tools:
            items.append(Text("No tools exposed.", style="dim"))
            continue

        table = Table(box=None, padding=(0, 1), expand=True, show_edge=False, header_style="dim")
        table.add_column("Tool", ratio=3, overflow="fold")
        table.add_column("Policy", width=7, no_wrap=True)
        table.add_column("Description", ratio=5, overflow="fold")
        for tool_index, tool in enumerate(tools):
            name = _terminal_text(tool["raw_name"])
            permission = _terminal_text(tool["permission"])
            description = _terminal_text(" ".join(tool["description"].split())) or "—"
            policy_style = "green" if permission == "allow" else "yellow"
            if narrow:
                if tool_index:
                    items.append(Text(""))
                heading = Text(name, overflow="fold")
                heading.append(f" [{permission}]", style=policy_style)
                items.extend([heading, Padding(Text(description, overflow="fold"), (0, 0, 0, 2))])
            else:
                table.add_row(Text(name), Text(permission, style=policy_style), Text(description))
        if not narrow:
            items.append(table)

    console.print(Panel(Group(*items), title=Text("MCP servers", style="bold"), title_align="left",
                        box=box.ROUNDED, border_style="dim", width=width, padding=(0, 1)))
