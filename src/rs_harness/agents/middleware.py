"""Official LangChain middleware used by the production Agent."""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolErrorMiddleware
from langchain_core.tools import ToolException

from ..config import Config
from ..inputs import validate_image_messages


def _tool_error(error: Exception, request) -> str | None:
    if isinstance(error, ToolException):
        return str(error)
    return None


class ImageInputMiddleware(AgentMiddleware):
    def __init__(self, config: Config):
        self.config = config

    async def awrap_model_call(self, request, handler):
        validate_image_messages(request.messages, self.config, getattr(request.model, "profile", None))
        return await handler(request)


def build_middleware(config: Config) -> list[AgentMiddleware]:
    """Build fresh middleware instances for one Agent graph."""

    return [
        ModelCallLimitMiddleware(
            run_limit=config.runtime.max_model_calls,
            exit_behavior="error",
        ),
        ToolErrorMiddleware(on_error=_tool_error),
        ImageInputMiddleware(config),
    ]
