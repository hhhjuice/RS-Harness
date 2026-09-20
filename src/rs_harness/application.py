"""Application service connecting sessions, Agent, Checkpointer, and records."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import HumanMessage

from .agents.main import build_agent
from .config import Config
from .context import AgentContext
from .inputs import build_human_message, extract_answer, summarize_message_content, validate_human_message, validate_image_messages
from .sessions.catalog import Session, SessionCatalog, SessionError
from .sessions.checkpoints import open_checkpointer
from .sessions.recording import TrajectoryRecorder, TrajectoryWriter


class InteractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    interaction_id: str
    status: str
    answer: str | None = None
    error: dict[str, str] | None = None
    session_status: str
    recording_errors: list[str] = Field(default_factory=list)


def create_session(config: Config, *, workdir: str | None = None) -> Session:
    return SessionCatalog.create(config, workdir=workdir)


async def ask(
    config: Config,
    session_id: str,
    text: str | HumanMessage,
    *,
    model: Any | None = None,
) -> InteractionResult:
    message = build_human_message(text) if isinstance(text, str) else text
    validate_human_message(message)
    validate_image_messages([message], config)
    interaction_id = uuid4().hex
    try:
        session = SessionCatalog.open(config, session_id)
    except SessionError as exc:
        return _failed_result(
            session_id=session_id,
            interaction_id=interaction_id,
            error=_session_error(exc),
            session_status="unknown",
        )

    writer = TrajectoryWriter(
        session.trajectory_path,
        session_id=session.session_id,
        interaction_id=interaction_id,
    )

    try:
        with session.interaction_lock():
            writer.append("user_input", {"content": summarize_message_content(message)})
            try:
                async with open_checkpointer(config) as checkpointer:
                    agent = build_agent(
                        config,
                        workdir=session.meta.workdir,
                        checkpointer=checkpointer,
                        model=model,
                    )
                    recorder = TrajectoryRecorder(writer)
                    result = await agent.ainvoke(
                        {"messages": [message]},
                        config={
                            "configurable": {"thread_id": session.session_id},
                            "callbacks": [recorder],
                            "metadata": {
                                "interaction_id": interaction_id,
                            },
                            "run_name": "rsh_agent",
                        },
                        context=AgentContext(
                            workdir=session.meta.workdir,
                            bash_timeout_s=config.runtime.bash_timeout_s,
                            max_tool_output_bytes=config.runtime.max_tool_output_bytes,
                        ),
                    )
                answer = extract_answer(result)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                writer.append("interaction_failed", error)
                session.save()
                return InteractionResult(
                    session_id=session.session_id,
                    interaction_id=interaction_id,
                    status="failed",
                    error=error,
                    session_status=session.meta.status,
                    recording_errors=writer.failures,
                )

            writer.append("interaction_succeeded", {"answer": answer})
            session.save()
            return InteractionResult(
                session_id=session.session_id,
                interaction_id=interaction_id,
                status="succeeded",
                answer=answer,
                session_status=session.meta.status,
                recording_errors=writer.failures,
            )
    except SessionError as exc:
        return _failed_result(
            session_id=session.session_id,
            interaction_id=interaction_id,
            error=_session_error(exc),
            session_status=session.meta.status,
        )


def _session_error(error: SessionError) -> dict[str, str]:
    code = str(error).split(":", 1)[0]
    return {"type": code, "message": str(error)}


def _failed_result(
    *,
    session_id: str,
    interaction_id: str,
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
