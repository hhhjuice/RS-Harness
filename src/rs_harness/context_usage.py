"""Offline accounting for the main agent's next model context."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict

from .config import Config, subagents_digest


CATEGORY_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool_schemas": "Tool schemas",
    "tool_results": "Tool results",
    "skills": "Skills",
    "images": "Images",
}
CONTEXT_SOURCE = "rsh_context_source"


class ContextCategory(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    tokens: int
    percent: float | None


class ContextUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    model: str
    context_window: int | None
    used_tokens: int
    remaining_tokens: int | None
    usage_percent: float | None
    categories: list[ContextCategory]
    estimation_method: Literal["approximate", "calibrated"]


def model_key(config: Config) -> str:
    """Bind calibration to a configured model and endpoint, without storing credentials."""
    identity = [config.model.model, (config.model.base_url or "https://api.openai.com/v1").rstrip("/")]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def _countable_message(message: BaseMessage) -> BaseMessage:
    # LangChain's approximate counter assumes list-content AI messages already
    # contain their tool calls. OpenAI can instead return text blocks plus a
    # separate tool_calls list; include those arguments exactly once as well.
    if isinstance(message, AIMessage) and isinstance(message.content, list) and message.tool_calls:
        embedded = set()
        for block in message.content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"function_call", "custom_tool_call"}:
                # Responses item IDs differ from the IDs in AIMessage.tool_calls.
                embedded.add(block.get("call_id"))
            elif block.get("type") in {"tool_call", "tool_use"}:
                embedded.add(block.get("id"))
        missing = [call for call in message.tool_calls if call.get("id") not in embedded]
        if missing:
            return message.model_copy(update={
                "content": [*message.content, {"type": "text", "text": repr(missing)}],
            })
    return message


def estimate_categories(system_message, messages, tools, *, skills_prompt: str = "") -> dict[str, int]:
    """Count the actual projected input; every token belongs to exactly one bucket."""
    counts = dict.fromkeys(CATEGORY_LABELS, 0)
    schemas = [convert_to_openai_tool(tool) for tool in tools]
    counts["tool_schemas"] = count_tokens_approximately([], tools=schemas)
    if system_message is not None:
        total = count_tokens_approximately([system_message])
        blocks = list(system_message.content_blocks)
        # SkillMiddleware appends one distinct block. Match that block, rather
        # than parsing headings or re-reading loaded skill files from disk.
        for index in range(len(blocks) - 1, -1, -1):
            if skills_prompt and blocks[index] == {"type": "text", "text": skills_prompt}:
                del blocks[index]
                break
        system_only = count_tokens_approximately([system_message.model_copy(update={"content": blocks})])
        counts["system"] = system_only
        counts["skills"] = total - system_only

    for original in messages:
        message = _countable_message(original)
        tokens = count_tokens_approximately([message])
        text_tokens = count_tokens_approximately([message], tokens_per_image=0)
        counts["images"] += tokens - text_tokens
        source = message.additional_kwargs.get(CONTEXT_SOURCE)
        if source == "system" or isinstance(message, SystemMessage):
            category = "system"
        elif source == "tool_results":
            category = "tool_results"
        elif isinstance(message, ToolMessage):
            category = "skills" if message.name == "load_skill" and message.status == "success" else "tool_results"
        elif isinstance(message, HumanMessage):
            category = "user"
        elif isinstance(message, AIMessage):
            category = "assistant"
        else:
            category = "system"
        counts[category] += text_tokens
    return counts


def resolve_context_window(config: Config) -> int | None:
    if config.model.context_window is not None:
        return config.model.context_window
    # Read the same bundled registry used by ChatOpenAI.profile. Constructing
    # ChatOpenAI just to inspect it would unnecessarily initialize HTTP clients
    # and require credentials/proxy dependencies for this offline command.
    try:
        from langchain_openai.data._profiles import _PROFILES
    except ImportError:
        return None
    window = _PROFILES.get(config.model.model, {}).get("max_input_tokens")
    return window if type(window) is int and window > 0 else None


def summarize_usage(config: Config, session_id: str, counts: dict[str, int], calibration=None) -> ContextUsage:
    factor = 1.0
    method = "approximate"
    if isinstance(calibration, dict) and calibration.get("model_key") == model_key(config):
        actual, estimated = calibration.get("input_tokens"), calibration.get("estimated_input_tokens")
        if type(actual) is int and actual > 0 and type(estimated) is int and estimated > 0:
            factor = actual / estimated
            method = "calibrated"
    window = resolve_context_window(config)
    categories = []
    for key, label in CATEGORY_LABELS.items():
        raw = counts[key]
        if key == "images" and not raw:
            continue
        tokens = round(raw * factor)
        categories.append(ContextCategory(
            key=key, label=label, tokens=tokens,
            percent=tokens / window * 100 if window else None,
        ))
    used = sum(item.tokens for item in categories)
    return ContextUsage(
        session_id=session_id, model=config.model.model, context_window=window,
        used_tokens=used, remaining_tokens=max(0, window - used) if window else None,
        usage_percent=used / window * 100 if window else None,
        categories=categories, estimation_method=method,
    )


async def get_context_usage(config: Config, session_id: str) -> ContextUsage:
    """Inspect persisted root context without running a graph, model, or tool."""
    from langgraph.graph.message import add_messages

    from .agents.main import build_agent_components
    from .agents.mcp import merge_connections
    from .sessions.catalog import SessionCatalog, SessionError
    from .sessions.checkpoints import open_checkpointer

    session = SessionCatalog.open(config, session_id)
    subagents = config.resolve_subagents()
    if subagents_digest(subagents) != session.meta.subagents_digest:
        raise SessionError("subagent_config_mismatch: restore the definitions or create a new session")
    state: dict[str, Any] = {}
    if config.runtime.checkpoint_file.is_file():
        async with open_checkpointer(config, read_only=True) as saver:
            saved = await saver.aget_tuple({"configurable": {"thread_id": session_id, "checkpoint_ns": ""}})
        if saved is not None:
            state = dict(saved.checkpoint["channel_values"])
            # Completed writes may await a sibling task or an interrupted step.
            # Apply only the root input channels using their actual reducers.
            for _, channel, value in saved.pending_writes or ():
                if channel == "messages":
                    state[channel] = add_messages(state.get(channel, []), value)
                elif channel == "mcp_connections":
                    state[channel] = merge_connections(state.get(channel, {}), value)
                elif channel == "context_calibration":
                    state[channel] = value
    components = build_agent_components(config, workdir=session.meta.workdir, subagents=subagents)
    request = components.context_request(state)
    counts = estimate_categories(
        request.system_message, request.messages, request.tools, skills_prompt=components.skills_prompt,
    )
    return summarize_usage(config, session_id, counts, state.get("context_calibration"))
