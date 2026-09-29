from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from pydantic import PrivateAttr


class ToolCallingChatModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ScriptedModel(FakeMessagesListChatModel):
    _call_count: int = PrivateAttr(default=0)
    _inputs: list[list[Any]] = PrivateAttr(default_factory=list)
    _bound_tools: list[Any] = PrivateAttr(default_factory=list)
    _tool_history: list[list[Any]] = PrivateAttr(default_factory=list)

    def __init__(self, responses, **kwargs):
        super().__init__(responses=responses, **kwargs)

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def inputs(self) -> list[list[Any]]:
        return self._inputs

    @property
    def bound_tools(self) -> list[Any]:
        return self._bound_tools

    @property
    def tool_history(self) -> list[list[Any]]:
        return self._tool_history

    def bind_tools(self, tools, **kwargs):
        self._bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._call_count += 1
        self._inputs.append(list(messages))
        self._tool_history.append(list(self._bound_tools))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
