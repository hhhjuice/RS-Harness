import json

from rs_harness.config import Config
from rs_harness.runtime.session import Session
from rs_harness.runtime.todos import (
    TodoManager,
    TodoItem,
    is_error_observation,
    run_todo_write,
)
from rs_harness.runtime.tools import TOOLS, TOOL_HANDLERS, dispatch_tool


def test_is_error_observation_matches_only_error_prefix():
    assert is_error_observation("Error: Max 20 todos allowed") is True
    assert is_error_observation("[ ] Fine\n\n(0/1 completed)") is False


def test_todo_write_replaces_manager_state():
    manager = TodoManager()
    result = run_todo_write(
        [{"content": "Inspect data", "status": "in_progress"}], manager
    )
    assert "[>] Inspect data" in result
    assert manager.items == [{"content": "Inspect data", "status": "in_progress"}]


def test_todo_write_empty_list_clears_manager():
    manager = TodoManager([{"content": "Old task"}])
    assert run_todo_write([], manager) == "No todos."
    assert manager.items == []


def test_render_marks_every_status_and_counts_completion():
    manager = TodoManager(
        [
            {"content": "First", "status": "completed"},
            {"content": "Second", "status": "in_progress"},
            {"content": "Third", "status": "pending"},
        ]
    )
    assert manager.render() == "[x] First\n[>] Second\n[ ] Third\n\n(1/3 completed)"


def test_update_accepts_todo_item_instances():
    manager = TodoManager()
    manager.update([TodoItem(content="Only item"), TodoItem(content="Done", status="completed")])
    assert manager.items == [
        {"content": "Only item", "status": "pending"},
        {"content": "Done", "status": "completed"},
    ]


def test_update_accepts_json_and_python_list_strings():
    manager = TodoManager()
    manager.update(json.dumps([{"content": "From JSON", "status": "pending"}]))
    assert manager.items == [{"content": "From JSON", "status": "pending"}]
    manager.update("[{'content': 'From Python', 'status': 'pending'}]")
    assert manager.items == [{"content": "From Python", "status": "pending"}]


def test_unparsable_string_input_is_rejected():
    manager = TodoManager([{"content": "Keep me"}])
    assert run_todo_write("not a list at all", manager) == (
        "Error: todos must be a list or JSON array string"
    )
    assert manager.items == [{"content": "Keep me", "status": "pending"}]


def test_non_list_input_is_rejected():
    manager = TodoManager([{"content": "Keep me"}])
    assert run_todo_write({"content": "dict"}, manager) == "Error: todos must be a list"
    assert manager.items == [{"content": "Keep me", "status": "pending"}]


def test_empty_content_is_rejected_and_keeps_previous_list():
    manager = TodoManager([{"content": "Old"}])
    assert run_todo_write([{"content": "   ", "status": "pending"}], manager) == (
        "Error: todos[0] requires content"
    )
    assert manager.items == [{"content": "Old", "status": "pending"}]


def test_non_string_content_is_rejected_and_keeps_previous_list():
    manager = TodoManager([{"content": "Old"}])
    for content in (None, 0, False):
        assert run_todo_write([{"content": content}], manager) == (
            "Error: todos[0] content must be a string"
        )
        assert manager.items == [{"content": "Old", "status": "pending"}]


def test_whitespace_content_on_todo_item_is_rejected_and_keeps_previous_list():
    manager = TodoManager([{"content": "Old"}])
    assert run_todo_write([TodoItem(content="   ")], manager) == (
        "Error: todos[0] requires content"
    )
    assert manager.items == [{"content": "Old", "status": "pending"}]


def test_invalid_status_is_rejected_and_keeps_previous_list():
    manager = TodoManager([{"content": "Old"}])
    assert run_todo_write(
        [{"content": "New", "status": "done"}, {"content": "Later"}], manager
    ) == "Error: todos[0] has invalid status 'done'"
    assert manager.items == [{"content": "Old", "status": "pending"}]


def test_non_object_item_is_rejected():
    assert run_todo_write(["plain string"], TodoManager()) == (
        "Error: todos[0] must be an object"
    )


def test_more_than_twenty_items_are_rejected():
    manager = TodoManager()
    twenty = [{"content": f"Task {i}"} for i in range(20)]
    manager.update(twenty)
    assert len(manager.items) == 20
    assert run_todo_write(twenty + [{"content": "Extra"}], manager) == (
        "Error: Max 20 todos allowed"
    )
    assert len(manager.items) == 20


def test_multiple_in_progress_items_are_rejected():
    manager = TodoManager([{"content": "Old"}])
    assert run_todo_write(
        [
            {"content": "One", "status": "in_progress"},
            {"content": "Two", "status": "in_progress"},
        ],
        manager,
    ) == "Error: Only one todo can be in_progress at a time"
    assert manager.items == [{"content": "Old", "status": "pending"}]


def test_todo_write_schema_is_registered_in_tools():
    todo = next(item for item in TOOLS if item["function"]["name"] == "todo_write")
    todos_schema = todo["function"]["parameters"]["properties"]["todos"]
    assert todos_schema["maxItems"] == 20
    assert todos_schema["items"]["properties"]["status"]["enum"] == [
        "pending", "in_progress", "completed"
    ]
    assert sorted(todos_schema["items"]["required"]) == ["content", "status"]
    assert TOOL_HANDLERS["todo_write"] is run_todo_write


def test_dispatch_todo_write_passes_optional_manager(tmp_path):
    manager = TodoManager()
    result = dispatch_tool(
        "todo_write", {"todos": [{"content": "Run checks"}]}, tmp_path,
        todo_manager=manager,
    )
    assert result.startswith("[ ] Run checks")
    assert manager.items[0]["content"] == "Run checks"


def test_dispatch_todo_write_without_manager_still_returns_render(tmp_path):
    result = dispatch_tool(
        "todo_write", {"todos": [{"content": "Standalone", "status": "pending"}]}, tmp_path
    )
    assert result == "[ ] Standalone\n\n(0/1 completed)"


def test_dispatch_todo_write_returns_error_observation(tmp_path):
    result = dispatch_tool(
        "todo_write", {"todos": [{"content": "", "status": "pending"}]}, tmp_path
    )
    assert result == "Error: todos[0] requires content"


def test_session_without_todos_field_is_reopened_with_empty_list(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("system\nWorking directory: {workdir}\n", encoding="utf-8")
    config = Config(
        model={"model": "test-model", "api_key_env": "UNUSED_RS_KEY"},
        loop={"max_turns": 2},
        runtime={
            "sessions_dir": tmp_path / "sessions",
            "system_prompt_file": prompt,
            "dotenv_file": tmp_path / "missing.env",
        },
    )
    session = Session.create(config, workdir=tmp_path)
    state_path = session.path / "session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state["todos"]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    reopened = Session.open(config, session.state.session_id)
    assert reopened.state.todos == []
    reopened.state.todos = [TodoItem(content="Resumed")]
    reopened.save_state()
    assert Session.open(config, session.state.session_id).state.todos == [
        TodoItem(content="Resumed")
    ]
