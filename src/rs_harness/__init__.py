"""RS-Harness public package."""

__version__ = "0.1.0"

from .agent_loop import AgentLoop, InputRequest, close_session, create_session, run
from .config import Config, ConfigError, LoopConfig, ModelConfig, RuntimeConfig, load_config
from .runtime.session import ErrorInfo, InteractionResult, SessionResult, SessionState, SessionStatus

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
    "load_config",
    "__version__",
]
