"""Minimal text-only LangChain agent assembly."""

from __future__ import annotations

from typing import Any

from ..config import Config


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
        checkpointer=checkpointer,
    )
