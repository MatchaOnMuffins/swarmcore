from __future__ import annotations

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentResult(BaseModel):
    agent_name: str
    input_task: str
    output: str
    model: str
    duration_seconds: float
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class SwarmResult(BaseModel):
    output: str
    context: dict[str, str]
    history: list[AgentResult]
