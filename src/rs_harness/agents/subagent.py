"""Configured LangChain subagents exposed through a single dispatch tool."""

from collections.abc import Sequence
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import merge_configs
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph import END, START, MessagesState, StateGraph

from ..config import Config, ResolvedSubagent
from ..context import AgentContext
from ..skills import SkillCatalog
from .middleware import build_middleware


def build_delegate_tool(
    config: Config, *, model: Any, tools: Sequence[BaseTool], workdir: str,
    subagents: Sequence[ResolvedSubagent], skills: SkillCatalog,
) -> BaseTool:
    from langchain.agents import create_agent

    available = {item.name: item for item in tools}
    skill_catalogs = {
        definition.id: skills.select(definition.skills, owner=f"Subagent {definition.id}")
        for definition in subagents
    }
    graphs = {}
    for definition in subagents:
        graph = create_agent(
            model=model,
            tools=[available[name] for name in definition.tools if name not in {"write_todos", "load_skill"}],
            system_prompt=(
                definition.system_prompt.replace("{workdir}", workdir)
                + f"\n\nIdentity: {definition.name} ({definition.id}). Workspace: {workdir}.\n"
                + f"Available tools: {', '.join(definition.tools) or 'none'}. Memory: {definition.persistence}.\n"
                + "You receive delegated requests, not the parent's conversation. File changes are shared. "
                "You cannot request human approval: file tools reject paths outside the workspace, "
                "and Bash rejects commands classified as requiring approval or forbidden. "
                "Bash still uses literal rules, not an OS sandbox. Never bypass a rejected action "
                "through another tool; report the limitation to the main agent. "
                "Only your final response is returned to the parent."
            ),
            middleware=build_middleware(
                config, tools=definition.tools, skills=skill_catalogs[definition.id], allow_approval=False,
            ),
            context_schema=AgentContext,
            checkpointer=True if definition.persistence == "per_thread" else None,
            name=f"rsh_{definition.id}",
        )
        if definition.persistence == "per_thread":
            node = f"subagent_{definition.id}"
            graph = (StateGraph(MessagesState).add_node(node, graph)
                     .add_edge(START, node).add_edge(node, END).compile())
        graphs[definition.id] = graph

    return make_delegate_tool(subagents, graphs)


def make_delegate_tool(subagents: Sequence[ResolvedSubagent], graphs: dict[str, Any]) -> BaseTool:
    """Share the actual dispatch schema with offline context inspection."""
    definitions = {item.id: item for item in subagents}

    @tool
    async def delegate_task(agent_id: str, query: str, runtime: ToolRuntime[AgentContext]):
        """Delegate a bounded task to a configured agent and return its final findings.

        Include the goal, necessary context, file scope and verification requirements.
        Agents share the workspace but cannot see this conversation or delegate further.
        For per_thread agents, reuse agent_id to follow up after its previous result.
        Call each per_thread identity at most once per batch. Only independent work can run together.
        """
        if agent_id not in graphs:
            raise ToolException("unknown_subagent: choose one of " + ", ".join(definitions))
        definition = definitions[agent_id]
        result = await graphs[agent_id].ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=merge_configs(runtime.config, {"metadata": {
                "subagent_id": agent_id, "subagent_name": definition.name,
            }}),
            context=runtime.context,
        )
        return result["messages"][-1].content

    delegate_task.description += "\nAvailable agents:\n" + "\n".join(
        f"- {item.id} ({item.name}): {item.description} Tools: {', '.join(item.tools) or 'none'}; memory: {item.persistence}."
        for item in subagents
    )
    return delegate_task
