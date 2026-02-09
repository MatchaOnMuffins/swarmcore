from __future__ import annotations

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    result: str = ""
    duration_seconds: float = 0.0


class LLMCallRecord(BaseModel):
    call_index: int
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_seconds: float = 0.0
    tool_calls_requested: list[str] = Field(default_factory=list)
    finish_reason: str = ""


class AgentResult(BaseModel):
    agent_name: str
    input_task: str
    output: str
    model: str
    duration_seconds: float
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    llm_call_count: int = 0
    tool_call_count: int = 0


class SwarmResult(BaseModel):
    output: str
    context: dict[str, str]
    history: list[AgentResult]
    duration_seconds: float = 0.0
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
