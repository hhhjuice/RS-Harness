"""Application service connecting sessions, Agent, Checkpointer, and records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from .agents.main import build_agent
from .config import Config, subagents_digest
from .context import AgentContext
from .inputs import build_human_message, extract_answer, summarize_message_content, validate_human_message, validate_image_messages
from .mcp import MCPRegistry
from .sessions.catalog import Session, SessionCatalog, SessionError
from .sessions.checkpoints import open_checkpointer
from .sessions.recording import TrajectoryRecorder, TrajectoryWriter


class InteractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    interaction_id: str | None
    status: Literal["succeeded", "failed", "interrupted"]
    answer: str | None = None
    error: dict[str, str] | None = None
    session_status: str
    interrupts: list[dict[str, Any]] = Field(default_factory=list)
    recording_errors: list[str] = Field(default_factory=list)


class ApprovalStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    session_status: str
    interaction_id: str | None
    interrupts: list[dict[str, Any]] = Field(default_factory=list)


class ApproveDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["approve"]


class RejectDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["reject"]
    message: str = ""


class ApprovalDecisions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decisions: list[Annotated[ApproveDecision | RejectDecision, Field(discriminator="type")]]


def validate_approval_decisions(decisions_by_interrupt: Any, interrupts: list[dict]) -> dict:
    batches = TypeAdapter(dict[str, ApprovalDecisions]).validate_python(decisions_by_interrupt, strict=True)
    if not batches or set(batches) != {item["id"] for item in interrupts}:
        raise ValueError("Decisions must match all current interrupt IDs exactly")
    for item in interrupts:
        decisions = batches[item["id"]].decisions
        if len(decisions) != len(item["action_requests"]):
            raise ValueError("Provide one decision per action, in its original order")
        for decision, review in zip(decisions, item["review_configs"], strict=True):
            if decision.type not in review["allowed_decisions"]:
                raise ValueError(f"Decision {decision.type!r} is not allowed")
    return {key: batch.model_dump(exclude_unset=True) for key, batch in batches.items()}


def create_session(config: Config, *, workdir: str | None = None) -> Session:
    return SessionCatalog.create(config, workdir=workdir)


async def ask(
    config: Config,
    session_id: str,
    text: str | HumanMessage,
    *,
    model: Any | None = None,
    callbacks: Sequence[BaseCallbackHandler] = (),
) -> InteractionResult:
    message = build_human_message(text) if isinstance(text, str) else text
    validate_human_message(message)
    validate_image_messages([message], config)
    return await _run_interaction(config, session_id, message=message, model=model, callbacks=callbacks)


async def get_pending_approvals(config: Config, session_id: str, *, model: Any | None = None) -> ApprovalStatus:
    session = SessionCatalog.open(config, session_id)
    async with open_checkpointer(config) as checkpointer:
        agent = build_agent(config, workdir=session.meta.workdir, checkpointer=checkpointer, model=model)
        snapshot = await agent.aget_state({"configurable": {"thread_id": session_id}})
    return _approval_status(session, snapshot)


async def get_configured_mcp(config: Config) -> list[dict[str, Any]]:
    """Discover tools from the configured MCP servers without creating or reading a session."""
    registry = MCPRegistry(config.mcp)
    servers = []
    for name in sorted(config.mcp.servers):
        catalog = await registry.connect(name, {})
        tools = []
        for alias, entry in sorted(catalog.entries.items()):
            schema = entry.tool.args_schema
            tools.append({
                "name": alias, "raw_name": entry.raw_name, "description": entry.tool.description,
                "input_schema": schema if isinstance(schema, dict) else schema.model_json_schema(),
                "permission": "confirm" if registry.requires_approval(entry) else "allow",
            })
        servers.append({"name": name, "transport": config.mcp.servers[name].transport, "tools": tools})
    return servers


async def resume(
    config: Config,
    session_id: str,
    decisions_by_interrupt: Any,
    *,
    model: Any | None = None,
    callbacks: Sequence[BaseCallbackHandler] = (),
) -> InteractionResult:
    return await _run_interaction(config, session_id, decisions_by_interrupt=decisions_by_interrupt, model=model, callbacks=callbacks)


async def _run_interaction(
    config: Config,
    session_id: str,
    *,
    message: HumanMessage | None = None,
    decisions_by_interrupt: Any = None,
    model: Any | None = None,
    callbacks: Sequence[BaseCallbackHandler] = (),
) -> InteractionResult:
    interaction_id = None
    session_status = "unknown"
    try:
        session = SessionCatalog.open(config, session_id)
        session_status = session.meta.status
        with session.interaction_lock():
            subagents = config.resolve_subagents()
            if subagents_digest(subagents) != session.meta.subagents_digest:
                raise SessionError(
                    "subagent_config_mismatch: Subagent definitions changed. Restore the original "
                    "ids, prompts, tools and persistence modes, or create a new session."
                )
            async with open_checkpointer(config) as checkpointer:
                agent = build_agent(
                    config, workdir=session.meta.workdir, checkpointer=checkpointer, model=model,
                    subagents=subagents,
                )
                snapshot = await agent.aget_state({"configurable": {"thread_id": session_id}})
                pending = _approval_status(session, snapshot)
                if message is not None:
                    if pending.interrupts:
                        return _approval_failure(pending, "session_pending_approval", "Resolve pending approvals before sending new input")
                    interaction_id = uuid4().hex
                    graph_input = {"messages": [message]}
                    event, payload = "user_input", {"content": summarize_message_content(message)}
                else:
                    if not pending.interrupts:
                        return _approval_failure(pending, "no_pending_approval", "This session has no pending approval")
                    if pending.interaction_id is None:
                        return _approval_failure(pending, "invalid_checkpoint_metadata", "Pending checkpoint has no valid interaction_id")
                    try:
                        decisions = validate_approval_decisions(decisions_by_interrupt, pending.interrupts)
                    except ValueError as exc:
                        return _approval_failure(pending, "invalid_approval_decisions", str(exc))
                    interaction_id = pending.interaction_id
                    graph_input = Command(resume=decisions)
                    event, payload = "approval_submitted", decisions
                writer = TrajectoryWriter(session.trajectory_path, session_id=session_id, interaction_id=interaction_id)
                writer.append(event, payload)
                return await _invoke_agent(config, session, agent, graph_input, writer, callbacks)
    except GraphBubbleUp:
        raise
    except SessionError as exc:
        return _failed_result(
            session_id=session_id,
            interaction_id=interaction_id,
            error=_session_error(exc),
            session_status=session.meta.status if session_status != "unknown" else "unknown",
        )
    except Exception as exc:
        return _failed_result(
            session_id=session_id, interaction_id=interaction_id,
            error={"type": type(exc).__name__, "message": str(exc)}, session_status=session_status,
        )


def _serialize_interrupts(interrupts) -> list[dict[str, Any]]:
    return [
        {"id": item.id, "action_requests": item.value["action_requests"], "review_configs": item.value["review_configs"]}
        for item in interrupts
    ]


def _approval_status(session: Session, snapshot) -> ApprovalStatus:
    interrupts = _serialize_interrupts(snapshot.interrupts)
    interaction_id = (snapshot.metadata or {}).get("interaction_id") if interrupts else None
    return ApprovalStatus(
        session_id=session.session_id, session_status=session.meta.status,
        interaction_id=interaction_id if isinstance(interaction_id, str) and interaction_id.strip() else None,
        interrupts=interrupts,
    )


def _approval_failure(pending: ApprovalStatus, code: str, message: str) -> InteractionResult:
    return InteractionResult(**pending.model_dump(), status="failed", error={"type": code, "message": message})


async def _invoke_agent(
    config: Config, session: Session, agent, graph_input, writer: TrajectoryWriter,
    callbacks: Sequence[BaseCallbackHandler] = (),
) -> InteractionResult:
    try:
        result = await agent.ainvoke(
            graph_input,
            config={
                "configurable": {"thread_id": session.session_id},
                "callbacks": [TrajectoryRecorder(writer), *callbacks],
                "metadata": {"interaction_id": writer.interaction_id},
                "run_name": "rsh_agent",
            },
            context=AgentContext(
                workdir=session.meta.workdir,
                bash_timeout_s=config.runtime.bash_timeout_s,
                max_tool_output_bytes=config.runtime.max_tool_output_bytes,
            ),
        )
        interrupts = _serialize_interrupts(result.get("__interrupt__", ()))
        if interrupts:
            outcome = {"status": "interrupted", "interrupts": interrupts}
            writer.append("interaction_interrupted", {"interrupts": interrupts})
        else:
            answer = extract_answer(result)
            outcome = {"status": "succeeded", "answer": answer}
            writer.append("interaction_succeeded", {"answer": answer})
    except GraphBubbleUp:
        raise
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        outcome = {"status": "failed", "error": error}
        writer.append("interaction_failed", error)
    session.save()
    return InteractionResult(
        session_id=session.session_id, interaction_id=writer.interaction_id,
        session_status=session.meta.status, recording_errors=writer.failures, **outcome,
    )


def _session_error(error: SessionError) -> dict[str, str]:
    code = str(error).split(":", 1)[0]
    return {"type": code, "message": str(error)}


def _failed_result(
    *,
    session_id: str,
    interaction_id: str | None,
    error: dict[str, str],
    session_status: str,
) -> InteractionResult:
    return InteractionResult(
        session_id=session_id,
        interaction_id=interaction_id,
        status="failed",
        error=error,
        session_status=session_status,
    )
