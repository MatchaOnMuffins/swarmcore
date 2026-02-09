from __future__ import annotations

import asyncio
import time

from swarmcore.context import SharedContext
from swarmcore.flow import Flow
from swarmcore.hooks import Event, EventType, Hooks
from swarmcore.models import AgentResult, SwarmResult, TokenUsage


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

            if isinstance(step, list):
                coros = [agent.run(task, context, hooks=hooks) for agent in step]
                results = await asyncio.gather(*coros)

                for result in results:
                    context.set(result.agent_name, result.output)
                    history.append(result)
            else:
                result = await step.run(task, context, hooks=hooks)
                context.set(result.agent_name, result.output)
                history.append(result)

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
