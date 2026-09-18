"""Minimal text-only LangChain agent assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents.middleware import build_middleware
from ..config import Config


@dataclass(frozen=True)
class AgentContext:
    """Per-invocation runtime dependencies kept outside Agent state."""

    workdir: str


def build_agent(
    config: Config,
    *,
    workdir: str,
    checkpointer: object | None = None,
    model: Any | None = None,
) -> Any:
    if model is None:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.model.model,
            "api_key": config.api_key(),
            "timeout": config.model.timeout_s,
            "max_tokens": config.model.max_tokens,
        }
        if config.model.base_url:
            kwargs["base_url"] = config.model.base_url
        model = ChatOpenAI(**kwargs)

    from langchain.agents import create_agent

    return create_agent(
        model=model,
        tools=[],
        system_prompt=config.system_prompt(workdir),
        middleware=build_middleware(config),
        context_schema=AgentContext,
        checkpointer=checkpointer,
    )
