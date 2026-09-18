"""Construction and extraction helpers for text messages."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def build_human_message(text: str) -> HumanMessage:
    if not text or not text.strip():
        raise ValueError("empty_input")
    return HumanMessage(content=text)


def extract_answer(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    if not messages:
        raise ValueError("agent_returned_no_messages")
    message = messages[-1]
    if not isinstance(message, AIMessage) and not hasattr(message, "content"):
        raise ValueError("agent_returned_invalid_message")
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)
