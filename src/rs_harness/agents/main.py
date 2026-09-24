"""LangChain agent assembly with local tools."""

from __future__ import annotations

from typing import Any

from ..agents.middleware import build_middleware
from ..config import Config
from ..context import AgentContext
from ..tools.filesystem import edit_file, glob, read_file, read_image, write_file
from ..tools.shell import bash


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
        tools=[read_file, write_file, edit_file, glob, bash, read_image],
        system_prompt=config.system_prompt(workdir),
        middleware=build_middleware(config),
        context_schema=AgentContext,
        checkpointer=checkpointer,
    )
