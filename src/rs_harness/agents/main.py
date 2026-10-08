"""LangChain agent assembly with local tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import ModelRequest, TodoListMiddleware
from langchain_core.messages import SystemMessage

from ..agents.middleware import ImageInputMiddleware, _project_image_tool_results, build_middleware
from ..agents.subagent import build_delegate_tool, make_delegate_tool
from ..config import Config, ResolvedSubagent
from ..context import AgentContext
from ..mcp import MCPRegistry
from ..skills import SkillLoader
from ..tools.filesystem import edit_file, glob, read_file, read_image, write_file
from ..tools.shell import bash
from .context_usage import ContextUsageMiddleware
from .mcp import MCPMiddleware
from .skills import SkillMiddleware


@dataclass
class AgentComponents:
    system_prompt: str
    tools: list[Any]
    middleware: list[Any]

    @property
    def skills_prompt(self) -> str:
        return next((item.skills_prompt for item in self.middleware if isinstance(item, SkillMiddleware)), "")

    def context_request(self, state: dict) -> ModelRequest:
        """Replay only pure input projections; never execute middleware hooks with I/O."""
        request = ModelRequest(
            model=None, messages=list(state.get("messages", [])), state=state,
            system_message=SystemMessage(content=self.system_prompt),
            tools=[*(tool for item in self.middleware for tool in getattr(item, "tools", ())), *self.tools],
        )
        for item in self.middleware:
            if isinstance(item, ImageInputMiddleware):
                request = request.override(messages=_project_image_tool_results(request.messages))
            elif isinstance(item, TodoListMiddleware):
                request = item.wrap_model_call(request, lambda projected: projected)
            elif isinstance(item, SkillMiddleware):
                request = item._with_skills(request)
            elif isinstance(item, MCPMiddleware):
                request = item.with_catalog(request, item.registry.saved_tools(state))
        return request


def build_agent_components(
    config: Config, *, workdir: str, model: Any | None = None,
    subagents: tuple[ResolvedSubagent, ...] | None = None,
) -> AgentComponents:
    """Assemble shared prompts/tools; model=None builds schemas without child graphs."""
    subagents = config.resolve_subagents() if subagents is None else subagents
    skills = SkillLoader(config.runtime.skills_dir)
    base_tools = [read_file, write_file, edit_file, glob, bash, read_image]
    tools = list(base_tools)
    if subagents:
        tools.append(
            make_delegate_tool(subagents, {}) if model is None else build_delegate_tool(
                config, model=model, tools=base_tools, workdir=workdir, subagents=subagents, skills=skills,
            )
        )
    mcp = MCPRegistry(config.mcp, local_names=[*(item.name for item in base_tools),
                      "delegate_task", "write_todos", "load_skill", "connect_mcp"],
                      max_output_bytes=config.runtime.max_tool_output_bytes)
    return AgentComponents(
        system_prompt=config.system_prompt(workdir), tools=tools,
        middleware=build_middleware(config, subagents=subagents, skills=skills, mcp=mcp),
    )


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

    components = build_agent_components(
        config, workdir=workdir, model=model if subagent_model is None else subagent_model, subagents=subagents,
    )
    # Only the root agent records calibration. Keep StopHook last and sample
    # after every middleware that transforms the model input.
    components.middleware.insert(-1, ContextUsageMiddleware(config, skills_prompt=components.skills_prompt))
    return create_agent(
        model=model,
        tools=components.tools,
        system_prompt=components.system_prompt,
        middleware=components.middleware,
        context_schema=AgentContext,
        checkpointer=checkpointer,
    )
