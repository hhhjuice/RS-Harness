"""Official LangChain middleware used by the production Agent."""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware

from ..config import Config


def build_middleware(config: Config) -> list[AgentMiddleware]:
    """Build fresh middleware instances for one Agent graph."""

    return [
        ModelCallLimitMiddleware(
            run_limit=config.runtime.max_model_calls,
            exit_behavior="error",
        ),
    ]
