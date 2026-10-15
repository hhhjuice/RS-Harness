"""Progressive skill disclosure through LangChain's model and tool hooks."""

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import SystemMessage
from langchain_core.tools import ToolException

from ..context import AgentContext
from ..skills import SkillCatalog


class SkillMiddleware(AgentMiddleware):
    def __init__(self, skills: SkillCatalog):
        self.skills_prompt = (
            f"\n\n## Available Skills\n\n{skills.catalog()}\n\n"
            "This catalog lists all registered skills for this Agent. Choose by description "
            "and load by the exact name here; no filesystem search for SKILL.md is needed. "
            "When a task needs a skill, call load_skill "
            "to read its full instructions before using it. Reuse already loaded instructions "
            "from the conversation when possible; unrelated tasks do not need a skill. "
            "Resolve relative resource paths against the returned Skill directory. "
            "Loading instructions does not execute actions or grant permissions. "
            "Use only your available tools and follow the existing permission rules."
        )

        @tool
        def load_skill(name: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Load a skill's complete instructions by a name from Available Skills."""
            try:
                skill = skills.load(name)
            except KeyError as exc:
                raise ToolException(f"Unknown skill {name!r}. Available skills: {', '.join(skills.skills) or 'none'}") from exc
            result = (
                f"Loaded skill: {skill.name}\nSkill directory: {skill.path.parent}\n"
                f"Skill file: {skill.path}\n\n{skill.content}"
            )
            if len(result.encode("utf-8")) > runtime.context.max_tool_output_bytes:
                raise ToolException(
                    f"Skill {name!r} exceeds max_tool_output_bytes; it was not loaded. "
                    "Split detailed instructions into reference files before loading again."
                )
            return result

        self.tools = [load_skill]

    def _with_skills(self, request):
        system = request.system_message or SystemMessage(content="")
        content = [*system.content_blocks, {"type": "text", "text": self.skills_prompt}]
        return request.override(system_message=system.model_copy(update={"content": content}))

    def wrap_model_call(self, request, handler):
        return handler(self._with_skills(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._with_skills(request))
