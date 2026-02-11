from __future__ import annotations

import sys
from typing import TextIO

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
    cost: float = 0.0


class AgentResult(BaseModel):
    agent_name: str
    input_task: str
    output: str
    summary: str = ""
    model: str
    duration_seconds: float
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    llm_call_count: int = 0
    tool_call_count: int = 0
    cost: float = 0.0


class SwarmResult(BaseModel):
    output: str
    context: dict[str, str]
    history: list[AgentResult]
    duration_seconds: float = 0.0
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    total_cost: float = 0.0

    def token_usage_table(self) -> str:
        """Format a per-agent token usage breakdown as a readable table."""
        if not self.history:
            return ""

        name_width = max(len(r.agent_name) for r in self.history)
        name_width = max(name_width, 5)  # minimum width for "TOTAL"

        show_cost = any(r.cost > 0 for r in self.history)

        lines: list[str] = []
        header = (
            f"  {'Agent':<{name_width}}  {'Tokens':>7}  "
            f"{'Calls':>5}  {'Tools':>5}  {'Duration':>8}"
        )
        rule_width = name_width + 32
        if show_cost:
            header += f"  {'Cost':>9}"
            rule_width += 11
        lines.append(header)
        rule = "  " + "\u2500" * rule_width
        lines.append(rule)

        for r in self.history:
            line = (
                f"  {r.agent_name:<{name_width}}  {r.token_usage.total_tokens:>7,}  "
                f"{r.llm_call_count:>5}  {r.tool_call_count:>5}  "
                f"{r.duration_seconds:>7.1f}s"
            )
            if show_cost:
                cost_str = f"${r.cost:.4f}"
                line += f"  {cost_str:>9}"
            lines.append(line)

        lines.append(rule)
        total_line = (
            f"  {'TOTAL':<{name_width}}  {self.total_token_usage.total_tokens:>7,}  "
            f"{'':>5}  {'':>5}  {self.duration_seconds:>7.1f}s"
        )
        if show_cost:
            total_cost_str = f"${self.total_cost:.4f}"
            total_line += f"  {total_cost_str:>9}"
        lines.append(total_line)
        return "\n".join(lines)

    def context_pull_report(self) -> str:
        """Analyze which agents pulled context from which others (pull mode).

        Returns an empty string if no agents used ``get_context``.
        """
        lines: list[str] = []
        name_width = max((len(r.agent_name) for r in self.history), default=5)

        for i, r in enumerate(self.history):
            pulls = [
                tc.arguments.get("agent_name", "?")
                for tc in r.tool_calls
                if tc.tool_name == "get_context"
            ]
            if not pulls:
                continue

            prior = [
                h.agent_name for h in self.history[:i] if h.agent_name != r.agent_name
            ]
            if not prior:
                continue

            tag = "SELECTIVE \u2713" if len(pulls) < len(prior) else "PULLED ALL \u26a0"
            skipped = [a for a in prior if a not in pulls]
            line = f"  {r.agent_name:<{name_width}}  {len(pulls)}/{len(prior)} pulled  {tag}"
            if skipped:
                line += f"  (skipped: {', '.join(skipped)})"
            lines.append(line)

        return "\n".join(lines)

    def summary(self) -> str:
        """Full post-run summary: token table + context pull analysis."""
        sections: list[str] = []

        token_table = self.token_usage_table()
        if token_table:
            sections.append(f"Token Usage\n{token_table}")

        pull_report = self.context_pull_report()
        if pull_report:
            sections.append(f"Context Pull Analysis\n{pull_report}")

        return "\n\n".join(sections)

    def print_summary(self, *, file: TextIO | None = None) -> None:
        """Print the full summary to a stream (default ``sys.stderr``)."""
        out = file or sys.stderr
        text = self.summary()
        if text:
            print(text, file=out)
