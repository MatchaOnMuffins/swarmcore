from __future__ import annotations

import asyncio
import re
import time
from typing import Callable, Literal

from swarmcore.agent import Agent
from swarmcore.context import SharedContext
from swarmcore.context_tools import make_context_tools
from swarmcore.flow import Flow
from swarmcore.hooks import (
    Event,
    EventType,
    Hooks,
    StepEndData,
    StepStartData,
    SwarmEndData,
    SwarmStartData,
)
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


def _collect_agent_names(step: Agent | list[Agent | Flow]) -> list[str]:
    """Collect agent names from a step for hook emission."""
    if isinstance(step, list):
        names: list[str] = []
        for item in step:
            if isinstance(item, Flow):
                names.extend(a.name for a in item.agents)
            else:
                names.append(item.name)
        return names
    return [step.name]


def _terminal_names(step: Agent | list[Agent | Flow]) -> set[str]:
    """Return the names of the terminal agents in a step."""
    if isinstance(step, list):
        names: set[str] = set()
        for item in step:
            if isinstance(item, Flow):
                last = item.steps[-1]
                if isinstance(last, list):
                    for sub in last:
                        if isinstance(sub, Flow):
                            names |= _terminal_names(sub.steps[-1])
                        else:
                            names.add(sub.name)
                else:
                    names.add(last.name)
            else:
                names.add(item.name)
        return names
    return {step.name}


class Swarm:
    def __init__(
        self,
        flow: Flow,
        hooks: Hooks | None = None,
        context_mode: Literal["push", "pull"] = "pull",
        timeout: float | None = None,
        max_retries: int | None = None,
        context_budget: int | None = None,
    ) -> None:
        self._steps = flow.steps
        self._agents = {a.name: a for a in flow.agents}
        self._hooks = hooks
        self._context_mode = context_mode
        self._timeout = timeout
        self._max_retries = max_retries
        self._context_budget = context_budget

    async def _run_agent_pull(
        self,
        agent: Agent,
        task: str,
        context: SharedContext,
        hooks: Hooks | None,
        prev_step_names: set[str] | None = None,
    ) -> AgentResult:
        """Run a single agent in pull mode, handling context tools and output parsing.

        The immediately preceding step's full output is pushed directly into the
        system prompt (no tool call needed).  Earlier agents are available via
        lightweight summaries and pull tools (``list_context``, ``get_context``,
        ``search_context``).
        """
        has_context = bool(context.keys())
        if has_context:
            prev_names = prev_step_names or set()
            entries = context.entries()

            # Split: previous-step outputs get pushed, earlier ones stay as pull
            prev_entries = [(n, s, f, c) for n, s, f, c in entries if n in prev_names]
            earlier_entries = [
                (n, s, f, c) for n, s, f, c in entries if n not in prev_names
            ]

            # Budget check: if prev-step outputs exceed budget, demote to pull
            if self._context_budget is not None and prev_entries:
                total_prev_chars = sum(c for _, _, _, c in prev_entries)
                if total_prev_chars > self._context_budget:
                    earlier_entries = prev_entries + earlier_entries
                    prev_entries = []

            hint_parts: list[str] = []

            # Push full output from immediately preceding agents
            for name, _summary, full, _count in prev_entries:
                hint_parts.append(f"## {name}\n{full}")

            # Summaries + pull tools for earlier agents
            if earlier_entries:
                hint_parts.append(
                    "\nEarlier agent outputs are also available. Use the "
                    "`list_context`, `get_context`, and `search_context` "
                    "tools to retrieve them as needed.\n"
                )
                for name, summary, _full, _count in earlier_entries:
                    hint_parts.append(f"- **{name}**: {summary}")

            context_hint: str | None = "\n".join(hint_parts) if hint_parts else None

            # Only inject pull tools when there are earlier entries to pull from
            extra_tools = make_context_tools(context) if earlier_entries else None
        else:
            context_hint = None
            extra_tools = None

        result = await agent.run(
            task,
            context,
            hooks=hooks,
            structured_output=True,
            extra_tools=extra_tools,
            context_hint=context_hint,
            swarm_timeout=self._timeout,
            swarm_max_retries=self._max_retries,
        )
        summary, detail = _parse_structured_output(result.output)
        result.output = detail
        result.summary = summary
        context.set(result.agent_name, detail, summary=summary)
        return result

    async def _run_agent_push(
        self,
        agent: Agent,
        task: str,
        context: SharedContext,
        hooks: Hooks | None,
        expand: set[str] | None,
        expand_tool: Callable[[str], str],
    ) -> AgentResult:
        """Run a single agent in push mode, handling expand tool and output parsing."""
        # Budget check: if expanded outputs exceed budget, demote all to summaries
        if self._context_budget is not None and expand:
            total_expand_chars = sum(len(context.get(name) or "") for name in expand)
            if total_expand_chars > self._context_budget:
                expand = set()

        context_keys = set(context.to_dict().keys())
        summarized = context_keys - (expand or set())
        extra_tools = [expand_tool] if summarized else None

        result = await agent.run(
            task,
            context,
            hooks=hooks,
            structured_output=True,
            expand=expand,
            extra_tools=extra_tools,
            swarm_timeout=self._timeout,
            swarm_max_retries=self._max_retries,
        )
        summary, detail = _parse_structured_output(result.output)
        result.output = detail
        result.summary = summary
        context.set(result.agent_name, detail, summary=summary)
        return result

    async def _run_subflow(
        self,
        subflow: Flow,
        task: str,
        context: SharedContext,
        hooks: Hooks | None,
        expand_tool: Callable[[str], str],
        prev_step_names: set[str],
    ) -> tuple[list[AgentResult], set[str]]:
        """Run a sub-flow's steps sequentially, returning results and terminal agent names."""
        results: list[AgentResult] = []
        sub_prev: set[str] = set(prev_step_names)

        for step in subflow.steps:
            if self._context_mode == "pull":
                if isinstance(step, list):
                    coros = []
                    for item in step:
                        if isinstance(item, Flow):
                            coros.append(
                                self._run_subflow(
                                    item, task, context, hooks, expand_tool, sub_prev
                                )
                            )
                        else:
                            coros.append(
                                self._run_agent_pull(
                                    item, task, context, hooks, sub_prev
                                )
                            )
                    gathered = await asyncio.gather(*coros)
                    current_names: set[str] = set()
                    for g in gathered:
                        if isinstance(g, tuple):
                            sub_results, sub_terminal = g
                            results.extend(sub_results)
                            current_names |= sub_terminal
                        else:
                            results.append(g)
                            current_names.add(g.agent_name)
                    sub_prev = current_names
                else:
                    result = await self._run_agent_pull(
                        step, task, context, hooks, sub_prev
                    )
                    results.append(result)
                    sub_prev = {result.agent_name}
            else:
                expand = sub_prev or None
                if isinstance(step, list):
                    coros = []
                    for item in step:
                        if isinstance(item, Flow):
                            coros.append(
                                self._run_subflow(
                                    item, task, context, hooks, expand_tool, sub_prev
                                )
                            )
                        else:
                            coros.append(
                                self._run_agent_push(
                                    item, task, context, hooks, expand, expand_tool
                                )
                            )
                    gathered = await asyncio.gather(*coros)
                    current_names = set()
                    for g in gathered:
                        if isinstance(g, tuple):
                            sub_results, sub_terminal = g
                            results.extend(sub_results)
                            current_names |= sub_terminal
                        else:
                            results.append(g)
                            current_names.add(g.agent_name)
                    sub_prev = current_names
                else:
                    result = await self._run_agent_push(
                        step, task, context, hooks, expand, expand_tool
                    )
                    results.append(result)
                    sub_prev = {result.agent_name}

        return results, sub_prev

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
                    SwarmStartData(task=task, step_count=len(self._steps)),
                )
            )

        prev_step_names: set[str] = set()
        expand_tool = _make_expand_tool(context)

        for step_index, step in enumerate(self._steps):
            if hooks and hooks.is_active:
                agent_names = _collect_agent_names(step)
                await hooks.emit(
                    Event(
                        EventType.STEP_START,
                        StepStartData(
                            step_index=step_index,
                            agents=agent_names,
                            parallel=isinstance(step, list),
                        ),
                    )
                )

            if self._context_mode == "pull":
                if isinstance(step, list):
                    coros = []
                    for item in step:
                        if isinstance(item, Flow):
                            coros.append(
                                self._run_subflow(
                                    item,
                                    task,
                                    context,
                                    hooks,
                                    expand_tool,
                                    prev_step_names,
                                )
                            )
                        else:
                            coros.append(
                                self._run_agent_pull(
                                    item, task, context, hooks, prev_step_names
                                )
                            )
                    gathered = await asyncio.gather(*coros)

                    current_step_names: set[str] = set()
                    for g in gathered:
                        if isinstance(g, tuple):
                            sub_results, sub_terminal = g
                            history.extend(sub_results)
                            current_step_names |= sub_terminal
                        else:
                            history.append(g)
                            current_step_names.add(g.agent_name)
                    prev_step_names = current_step_names
                else:
                    result = await self._run_agent_pull(
                        step, task, context, hooks, prev_step_names
                    )
                    history.append(result)
                    prev_step_names = {result.agent_name}
            else:
                # Push mode
                expand = prev_step_names or None

                if isinstance(step, list):
                    coros = []
                    for item in step:
                        if isinstance(item, Flow):
                            coros.append(
                                self._run_subflow(
                                    item,
                                    task,
                                    context,
                                    hooks,
                                    expand_tool,
                                    prev_step_names,
                                )
                            )
                        else:
                            coros.append(
                                self._run_agent_push(
                                    item, task, context, hooks, expand, expand_tool
                                )
                            )
                    gathered = await asyncio.gather(*coros)

                    current_step_names_push: set[str] = set()
                    for g in gathered:
                        if isinstance(g, tuple):
                            sub_results, sub_terminal = g
                            history.extend(sub_results)
                            current_step_names_push |= sub_terminal
                        else:
                            history.append(g)
                            current_step_names_push.add(g.agent_name)
                    prev_step_names = current_step_names_push
                else:
                    result = await self._run_agent_push(
                        step, task, context, hooks, expand, expand_tool
                    )
                    history.append(result)
                    prev_step_names = {result.agent_name}

            if hooks and hooks.is_active:
                await hooks.emit(
                    Event(
                        EventType.STEP_END,
                        StepEndData(step_index=step_index),
                    )
                )

        swarm_duration = round(time.monotonic() - swarm_start, 3)

        total_usage = TokenUsage()
        total_cost = 0.0
        for agent_result in history:
            total_usage.prompt_tokens += agent_result.token_usage.prompt_tokens
            total_usage.completion_tokens += agent_result.token_usage.completion_tokens
            total_usage.total_tokens += agent_result.token_usage.total_tokens
            total_cost += agent_result.cost

        swarm_result = SwarmResult(
            output=history[-1].output,
            context=context.to_dict(),
            history=history,
            duration_seconds=swarm_duration,
            total_token_usage=total_usage,
            total_cost=total_cost,
        )

        if hooks and hooks.is_active:
            await hooks.emit(
                Event(
                    EventType.SWARM_END,
                    SwarmEndData(
                        duration_seconds=swarm_duration,
                        agent_count=len(history),
                        total_cost=total_cost,
                    ),
                )
            )

        return swarm_result
