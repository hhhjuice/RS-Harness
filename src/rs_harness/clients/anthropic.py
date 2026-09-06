"""Anthropic SDK integration for the v0.1.0 loop."""

from __future__ import annotations

from typing import Any

from ..config import Config, ModelConfig


def create_client(config: Config) -> Any:
    api_key = config.api_key()
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("Anthropic SDK is not installed") from exc

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": config.model.timeout_s,
    }
    if config.model.base_url:
        kwargs["base_url"] = config.model.base_url
    return Anthropic(**kwargs)


def create_message(
    client: Any,
    model: ModelConfig,
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> Any:
    """Call Anthropic's native Messages API with no tools registered."""

    return client.messages.create(
        model=model.model,
        system=system_prompt,
        messages=messages,
        max_tokens=model.max_tokens,
        tools=[],
    )
