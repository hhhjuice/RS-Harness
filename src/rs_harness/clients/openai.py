"""OpenAI SDK integration for the v0.1.0 loop."""

from __future__ import annotations

from typing import Any

from ..config import Config, ModelConfig


def create_client(config: Config) -> Any:
    api_key = config.api_key()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is not installed") from exc

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": config.model.timeout_s,
    }
    if config.model.base_url:
        kwargs["base_url"] = config.model.base_url
    return OpenAI(**kwargs)


def create_completion(
    client: Any,
    model: ModelConfig,
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> Any:
    """Call OpenAI's native Chat Completions API with no tools registered."""

    request_messages = [{"role": "system", "content": system_prompt}, *messages]
    return client.chat.completions.create(
        model=model.model,
        messages=request_messages,
        max_tokens=model.max_tokens,
        tools=[],
    )
