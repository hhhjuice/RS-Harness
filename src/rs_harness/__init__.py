"""RS-Harness public package."""

__version__ = "0.1.0"

from .agent_loop import AgentLoop, InputRequest, close_session, create_session, run
from .config import Config, ConfigError, LoopConfig, ModelConfig, RuntimeConfig
from .runtime.session import ErrorInfo, InteractionResult, SessionResult, SessionState, SessionStatus
from .runtime.tools import (
    TOOLS,
    TOOL_HANDLERS,
    dispatch_tool,
    normalize_workdir,
    run_bash,
    run_edit,
    run_glob,
    run_read,
    run_write,
    safe_path,
)

__all__ = [
    "Config",
    "ConfigError",
    "LoopConfig",
    "ModelConfig",
    "RuntimeConfig",
    "ErrorInfo",
    "InteractionResult",
    "AgentLoop",
    "InputRequest",
    "SessionResult",
    "SessionState",
    "SessionStatus",
    "run",
    "create_session",
    "close_session",
    "TOOLS",
    "TOOL_HANDLERS",
    "dispatch_tool",
    "normalize_workdir",
    "safe_path",
    "run_bash",
    "run_read",
    "run_write",
    "run_edit",
    "run_glob",
    "__version__",
]
