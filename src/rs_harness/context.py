"""Trusted dependencies injected for each Agent invocation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentContext:
    workdir: str
    bash_timeout_s: float = 60
    max_tool_output_bytes: int = 65536
