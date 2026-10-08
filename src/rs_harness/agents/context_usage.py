"""Record one root-model calibration sample at the final request boundary."""

from typing import Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ExtendedModelResponse
from langchain_core.messages import AIMessage
from langgraph.types import Command

from ..config import Config
from ..context_usage import estimate_categories, model_key


class ContextUsageState(AgentState):
    context_calibration: NotRequired[dict[str, Any] | None]


class ContextUsageMiddleware(AgentMiddleware):
    state_schema = ContextUsageState

    def __init__(self, config: Config, *, skills_prompt: str):
        self.model_key = model_key(config)
        self.skills_prompt = skills_prompt

    def _with_calibration(self, request, response):
        result = response.model_response if isinstance(response, ExtendedModelResponse) else response
        messages = [result] if isinstance(result, AIMessage) else result.result
        message = next((item for item in reversed(messages) if isinstance(item, AIMessage)), None)
        actual = (message.usage_metadata or {}).get("input_tokens") if message is not None else None
        calibration = None
        if type(actual) is int and actual > 0:
            counts = estimate_categories(
                request.system_message, request.messages, request.tools, skills_prompt=self.skills_prompt,
            )
            calibration = {
                "model_key": self.model_key,
                "estimated_input_tokens": sum(counts.values()),
                "input_tokens": actual,
            }
        # This middleware is innermost; the handler returns the model response.
        # Preserve any existing state update if another inner wrapper is added.
        update = dict(response.command.update or {}) if isinstance(response, ExtendedModelResponse) and response.command else {}
        update["context_calibration"] = calibration
        if isinstance(result, AIMessage):
            from langchain.agents.middleware import ModelResponse
            result = ModelResponse(result=[result])
        return ExtendedModelResponse(model_response=result, command=Command(update=update))

    def wrap_model_call(self, request, handler):
        return self._with_calibration(request, handler(request))

    async def awrap_model_call(self, request, handler):
        return self._with_calibration(request, await handler(request))
