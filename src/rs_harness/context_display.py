"""Compact, responsive terminal rendering for /context."""

import os
import sys

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .context_usage import ContextUsage


def print_context_usage(usage: ContextUsage, *, console: Console | None = None) -> None:
    console = console or Console(highlight=False, force_terminal=sys.stdout.isatty(),
                                 color_system=None if "NO_COLOR" in os.environ else "auto")
    width = min(console.width, 76)
    compact = width < 40
    narrow = width < 56
    inner_width = max(1, width - (0 if compact else 4))
    percent = f"{usage.usage_percent:.1f}%" if usage.usage_percent is not None else "—"
    window = f"{usage.context_window:,}" if usage.context_window is not None else "—"
    free = f"{usage.remaining_tokens:,}" if usage.remaining_tokens is not None else "—"
    color = "red" if (usage.usage_percent or 0) >= 90 else "yellow" if (usage.usage_percent or 0) >= 60 else "blue"
    items = []
    if narrow:
        items.append(Text(f"≈ {usage.used_tokens:,} tokens · {percent}", style="bold"))
        items.append(Text(f"Window {window} · Free {free}", style="dim"))
    else:
        summary = Table.grid(expand=True)
        summary.add_column(ratio=1)
        summary.add_column(justify="right")
        summary.add_row(Text(f"≈ {usage.used_tokens:,} / {window} tokens", style="bold"), Text(percent, style=f"bold {color}"))
        items.append(summary)
    if usage.usage_percent is not None:
        filled = min(inner_width, max(0, round(inner_width * usage.usage_percent / 100)))
        bar = Text("█" * filled, style=color)
        bar.append("░" * (inner_width - filled), style="bright_black")
        items.append(bar)
    if not narrow:
        items.append(Text(f"Free {free}", style="dim"))
    items.append(Text(""))
    table = Table(box=None, padding=(0, 1), expand=True, show_edge=False)
    table.add_column("Category", ratio=1, no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Window", justify="right", no_wrap=True)
    short_labels = {"tool_schemas": "Schemas", "tool_results": "Results", "assistant": "Assist."}
    for category in usage.categories:
        label = short_labels.get(category.key, category.label) if narrow else category.label
        table.add_row(label, f"{category.tokens:,}", f"{category.percent:.1f}%" if category.percent is not None else "—")
    items.append(table)
    if usage.context_window is None:
        items.extend([Text(""), Text("Set model.context_window", style="dim")])
    safe_model = "".join(char if char.isprintable() else char.encode("unicode_escape").decode("ascii") for char in usage.model)
    title = Text(f"Context · {safe_model}", style="bold", no_wrap=True, overflow="ellipsis")
    title.truncate(max(1, width - 6), overflow="ellipsis")
    if compact:
        console.print(Group(title, *items))
    else:
        console.print(Panel(Group(*items), title=title, title_align="left", box=box.ROUNDED,
                            border_style="dim", width=width, padding=(0, 1)))
