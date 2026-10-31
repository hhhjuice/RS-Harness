"""One independent Python process for checkpoint and model acceptance tests."""

import asyncio
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.middleware import AgentMiddleware

from rs_harness.agents.main import build_agent
from rs_harness.application import ask, create_session, get_pending_approvals, resume
from rs_harness.config import Config, MCPConfig
from rs_harness.context import AgentContext
from rs_harness.mcp import MCPRecoveryError
from rs_harness.sessions.checkpoints import open_checkpointer
from tests.models import ScriptedModel


async def main():
    request = json.load(sys.stdin)
    config = Config.load(sys.argv[1])
    if "mcp" in request:
        config = config.model_copy(update={"mcp": MCPConfig.model_validate(request["mcp"])})
    if "runtime" in request:
        config = config.model_copy(update={"runtime": config.runtime.model_copy(update={
            key: Path(value) if key.endswith(("_dir", "_file")) else value for key, value in request["runtime"].items()
        })})
    if request.get("live"):
        model = None
    else:
        model = ScriptedModel([AIMessage(**response) for response in request.get("responses", [{"content": "done"}])])
    tool_sets = []
    if request.get("live"):
        from rs_harness.agents import main as main_agent, subagent

        class FixtureGuard(AgentMiddleware):
            def __init__(self, child):
                self.child = child

            async def awrap_model_call(self, request, handler):
                tool_sets.append({"child": self.child, "names": [item.name for item in request.tools]})
                return await handler(request)

            async def awrap_tool_call(self, request, handler):
                allowed = {"connect_mcp", "read_mcp_resource", "get_mcp_prompt",
                           "read_file", "load_skill", "write_todos", "delegate_task",
                           "mcp__docs__search", "mcp__docs__get_version", "mcp__deploy__status",
                           "mcp__deploy__trigger", "mcp__deploy__unlisted", "mcp__all__picture"}
                if request.tool_call["name"] not in allowed:
                    raise AssertionError("Live fixture guard refused tool: " + request.tool_call["name"])
                return await handler(request)

        original = main_agent.build_middleware

        def guarded(*args, **kwargs):
            return [*original(*args, **kwargs), FixtureGuard(not kwargs.get("allow_approval", True))]

        main_agent.build_middleware = subagent.build_middleware = guarded
    session_id = request.get("session_id") or create_session(config, workdir=request["workdir"]).session_id
    action = request.get("action", "ask")
    if action in {"tool_pause", "tool_resume"}:
        async with open_checkpointer(config) as saver:
            graph = build_agent(config, workdir=request["workdir"], model=model, checkpointer=saver)
            try:
                await graph.ainvoke({"messages": [HumanMessage("resume probe")]} if action == "tool_pause" else None,
                                   {"configurable": {"thread_id": session_id}},
                                   context=AgentContext(request["workdir"]),
                                   interrupt_before=["tools"] if action == "tool_pause" else None)
                result = {"session_id": session_id, "status": "succeeded"}
            except MCPRecoveryError as exc:
                result = {"session_id": session_id, "status": "failed", "error": str(exc)}
    elif action == "pending":
        result = await get_pending_approvals(config, session_id, model=model)
    elif action == "resume":
        result = await resume(config, session_id, request["decisions"], model=model)
    else:
        result = await ask(config, session_id, request.get("text", "run"), model=model)
    async with open_checkpointer(config) as saver:
        graph = build_agent(config, workdir=request["workdir"], model=model, checkpointer=saver)
        state = await graph.aget_state({"configurable": {"thread_id": session_id}})
    print(json.dumps({"result": result if isinstance(result, dict) else result.model_dump(mode="json"), "model": config.model.model,
                      "tool_sets": tool_sets,
                      "connections": state.values.get("mcp_connections", {}),
                      "messages": [item.model_dump(mode="json") for item in state.values.get("messages", [])],
                      "tool_history": [[item.name for item in batch] for batch in model.tool_history] if model else []}))


if __name__ == "__main__":
    asyncio.run(main())
