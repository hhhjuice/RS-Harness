"""Preserve reasoning returned by OpenAI-compatible chat endpoints."""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI


def _preserve_reasoning(message: AIMessage, payload: dict[str, Any]) -> None:
    for key in ("reasoning_content", "reasoning"):
        value = payload.get(key)
        if isinstance(value, str):
            message.additional_kwargs[key] = value


class ReasoningChatOpenAI(ChatOpenAI):
    """Keep provider reasoning separate from answer text and tool calls."""

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if generation is not None and isinstance(generation.message, AIMessage) and choices:
            _preserve_reasoning(generation.message, choices[0].get("delta") or {})
        return generation

    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        payload = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, payload["choices"]):
            if isinstance(generation.message, AIMessage):
                _preserve_reasoning(generation.message, choice["message"])
        return result
