"""LangChain agent assembly with local tools."""

from __future__ import annotations

from typing import Any

from ..agents.middleware import build_middleware
from ..agents.subagent import build_delegate_tool
from ..config import Config, ResolvedSubagent
from ..context import AgentContext
from ..skills import SkillLoader
from ..tools.filesystem import edit_file, glob, read_file, read_image, write_file
from ..tools.shell import bash


def build_agent(
    config: Config,
    *,
    workdir: str,
    checkpointer: object | None = None,
    model: Any | None = None,
    subagent_model: Any | None = None,
    subagents: tuple[ResolvedSubagent, ...] | None = None,
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
        if config.model.reasoning_effort is not None:
            kwargs["reasoning_effort"] = config.model.reasoning_effort
        model = ChatOpenAI(**kwargs)

    from langchain.agents import create_agent

    subagents = config.resolve_subagents() if subagents is None else subagents
    skills = SkillLoader(config.runtime.skills_dir)
    base_tools = [read_file, write_file, edit_file, glob, bash, read_image]
    delegate_task = build_delegate_tool(
        config, model=model if subagent_model is None else subagent_model,
        tools=base_tools, workdir=workdir, subagents=subagents, skills=skills,
    )
    return create_agent(
        model=model,
        tools=[*base_tools, delegate_task],
        system_prompt=config.system_prompt(workdir),
        middleware=build_middleware(config, subagents=subagents, skills=skills),
        context_schema=AgentContext,
        checkpointer=checkpointer,
    )
