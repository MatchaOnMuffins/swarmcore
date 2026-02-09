from __future__ import annotations

import asyncio
import re
import time
from typing import Callable

from swarmcore.context import SharedContext
from swarmcore.flow import Flow
from swarmcore.hooks import Event, EventType, Hooks
from swarmcore.models import AgentResult, SwarmResult, TokenUsage

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


def _parse_structured_output(output: str) -> tuple[str, str]:
    """Extract summary and detail from structured agent output.

    Returns ``(summary, detail)`` where *detail* is the output with
    the ``<summary>`` block removed.  If no tags are found, the full
    output is used for both summary and detail (graceful degradation).
    """
    match = _SUMMARY_RE.search(output)
    if not match:
        return output, output
    summary = match.group(1).strip()
    detail = (output[: match.start()] + output[match.end() :]).strip()
    return summary, detail


def _make_expand_tool(context: SharedContext) -> Callable[[str], str]:
    """Create an ``expand_context`` tool bound to the given context."""

    def expand_context(agent_name: str) -> str:
        """Retrieve the full detailed output from a prior agent when its
        summary is not sufficient. Call this when you need to see the
        complete original output rather than just the summary shown in
        your context.

        agent_name: The name of the prior agent whose full output you want
        """
        full = context.get(agent_name)
        if full is None:
            return f"No context found for agent '{agent_name}'."
        return full

    return expand_context


class Swarm:
    def __init__(self, flow: Flow, hooks: Hooks | None = None) -> None:
        self._steps = flow.steps
        self._agents = {a.name: a for a in flow.agents}
        self._hooks = hooks

    async def run(self, task: str) -> SwarmResult:
        """Execute the swarm workflow on the given task."""
        swarm_start = time.monotonic()
        context = SharedContext()
        history: list[AgentResult] = []
        hooks = self._hooks

        if hooks and hooks.is_active:
            await hooks.emit(
                Event(
                    EventType.SWARM_START,
                    {"task": task, "step_count": len(self._steps)},
                )
            )

        prev_step_names: set[str] = set()
        expand_tool = _make_expand_tool(context)

        for step_index, step in enumerate(self._steps):
            if hooks and hooks.is_active:
                if isinstance(step, list):
                    agent_names = [a.name for a in step]
                else:
                    agent_names = [step.name]
                await hooks.emit(
                    Event(
                        EventType.STEP_START,
                        {
                            "step_index": step_index,
                            "agents": agent_names,
                            "parallel": isinstance(step, list),
                        },
                    )
                )

            expand = prev_step_names or None

            # Offer the expand tool when there are summarized entries
            context_keys = set(context.to_dict().keys())
            summarized = context_keys - (expand or set())
            extra_tools = [expand_tool] if summarized else None

            if isinstance(step, list):
                coros = [
                    agent.run(
                        task,
                        context,
                        hooks=hooks,
                        structured_output=True,
                        expand=expand,
                        extra_tools=extra_tools,
                    )
                    for agent in step
                ]
                results = await asyncio.gather(*coros)

                current_step_names: set[str] = set()
                for result in results:
                    summary, detail = _parse_structured_output(result.output)
                    result.output = detail
                    result.summary = summary
                    context.set(result.agent_name, detail, summary=summary)
                    history.append(result)
                    current_step_names.add(result.agent_name)
                prev_step_names = current_step_names
            else:
                result = await step.run(
                    task,
                    context,
                    hooks=hooks,
                    structured_output=True,
                    expand=expand,
                    extra_tools=extra_tools,
                )
                summary, detail = _parse_structured_output(result.output)
                result.output = detail
                result.summary = summary
                context.set(result.agent_name, detail, summary=summary)
                history.append(result)
                prev_step_names = {result.agent_name}

            if hooks and hooks.is_active:
                await hooks.emit(
                    Event(
                        EventType.STEP_END,
                        {"step_index": step_index},
                    )
                )

        swarm_duration = round(time.monotonic() - swarm_start, 3)

        total_usage = TokenUsage()
        for agent_result in history:
            total_usage.prompt_tokens += agent_result.token_usage.prompt_tokens
            total_usage.completion_tokens += agent_result.token_usage.completion_tokens
            total_usage.total_tokens += agent_result.token_usage.total_tokens

        swarm_result = SwarmResult(
            output=history[-1].output,
            context=context.to_dict(),
            history=history,
            duration_seconds=swarm_duration,
            total_token_usage=total_usage,
        )

        if hooks and hooks.is_active:
            await hooks.emit(
                Event(
                    EventType.SWARM_END,
                    {
                        "duration_seconds": swarm_duration,
                        "agent_count": len(history),
                    },
                )
            )

        return swarm_result
