"""RS-Harness public package."""

__version__ = "0.2.2"

from .agent_loop import InputRequest, close_session, create_session, run
from .config import Config, ConfigError, LoopConfig, ModelConfig, RuntimeConfig
from .runtime.session import ErrorInfo, ExecutionStatus, InteractionResult, SessionResult, SessionState, SessionStatus
from .runtime.trace import EventLedger
from .runtime.todos import TodoItem, TodoManager, run_todo_write
from .runtime.hooks import HookContext, HookDecision, HookEvent, HookRegistry, PermissionCallback, PermissionRequest, check_permission, default_hooks, log_hook, todo_display_hook, tool_error_hook
from .runtime.tools import (
    TOOLS,
    SUB_TOOLS,
    TOOL_HANDLERS,
    dispatch_tool,
    normalize_workdir,
    run_bash,
    run_edit,
    run_glob,
    run_read,
    run_read_image,
    run_task,
    run_write,
    safe_path,
)
from .runtime.background import (
    BackgroundTaskManager,
    TaskSnapshot,
    get_background_task_manager,
)

__all__ = [
    "Config",
    "ConfigError",
    "LoopConfig",
    "ModelConfig",
    "RuntimeConfig",
    "ErrorInfo",
    "InteractionResult",
    "InputRequest",
    "SessionResult",
    "SessionState",
    "SessionStatus",
    "ExecutionStatus",
    "run",
    "create_session",
    "close_session",
    "TOOLS",
    "SUB_TOOLS",
    "TOOL_HANDLERS",
    "dispatch_tool",
    "normalize_workdir",
    "safe_path",
    "run_bash",
    "run_read",
    "run_write",
    "run_edit",
    "run_glob",
    "run_read_image",
    "run_task",
    "BackgroundTaskManager",
    "TaskSnapshot",
    "get_background_task_manager",
    "HookContext",
    "HookDecision",
    "HookEvent",
    "HookRegistry",
    "PermissionCallback",
    "PermissionRequest",
    "check_permission",
    "default_hooks",
    "log_hook",
    "todo_display_hook",
    "tool_error_hook",
    "EventLedger",
    "TodoItem",
    "TodoManager",
    "run_todo_write",
    "__version__",
]
