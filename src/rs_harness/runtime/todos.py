"""Planning state for the ``todo_write`` tool."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    status: TodoStatus = "pending"


class TodoManager:
    """Own a complete todo list and replace it atomically after validation."""

    def __init__(self, todos: Iterable[TodoItem | Mapping[str, Any]] | None = None):
        self.items: list[dict[str, str]] = []
        if todos is not None:
            self.update(list(todos))

    def update(self, todos: list[TodoItem | Mapping[str, Any]] | str) -> str:
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                try:
                    todos = ast.literal_eval(todos)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError("todos must be a list or JSON array string") from exc
        if not isinstance(todos, list):
            raise ValueError("todos must be a list")
        if len(todos) > 20:
            raise ValueError("Max 20 todos allowed")
        validated: list[dict[str, str]] = []
        in_progress = 0
        for index, todo in enumerate(todos):
            if isinstance(todo, TodoItem):
                content, status = todo.content.strip(), todo.status
            elif isinstance(todo, Mapping):
                raw_content = todo.get("content", "")
                if not isinstance(raw_content, str):
                    raise ValueError(f"todos[{index}] content must be a string")
                content = raw_content.strip()
                status = str(todo.get("status", "pending")).lower()
            else:
                raise ValueError(f"todos[{index}] must be an object")
            if not content:
                raise ValueError(f"todos[{index}] requires content")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"todos[{index}] has invalid status '{status}'")
            if status == "in_progress":
                in_progress += 1
            validated.append({"content": content, "status": status})
        if in_progress > 1:
            raise ValueError("Only one todo can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        lines = [f"{markers[item['status']]} {item['content']}" for item in self.items]
        done = sum(item["status"] == "completed" for item in self.items)
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


def run_todo_write(
    todos: list[TodoItem | Mapping[str, Any]] | str, manager: TodoManager | None = None
) -> str:
    """Validate and store a complete todo list, returning a concise observation."""

    target = manager or TodoManager()
    try:
        return target.update(todos)
    except ValueError as exc:
        return f"Error: {exc}"


def is_error_observation(output: str) -> bool:
    """Whether a tool observation is a rejected update rather than a success."""

    return output.startswith("Error")


__all__ = ["TodoItem", "TodoManager", "TodoStatus", "run_todo_write", "is_error_observation"]
