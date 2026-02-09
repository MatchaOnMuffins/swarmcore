from __future__ import annotations

import asyncio

from swarmcore.context import SharedContext
from swarmcore.flow import Flow
from swarmcore.models import AgentResult, SwarmResult


class Swarm:
    def __init__(self, flow: Flow) -> None:
        self._steps = flow.steps
        self._agents = {a.name: a for a in flow.agents}

    async def run(self, task: str) -> SwarmResult:
        """Execute the swarm workflow on the given task."""
        context = SharedContext()
        history: list[AgentResult] = []

        for step in self._steps:
            if isinstance(step, list):
                coros = [agent.run(task, context) for agent in step]
                results = await asyncio.gather(*coros)

                for result in results:
                    context.set(result.agent_name, result.output)
                    history.append(result)
            else:
                result = await step.run(task, context)
                context.set(result.agent_name, result.output)
                history.append(result)

        return SwarmResult(
            output=history[-1].output,
            context=context.to_dict(),
            history=history,
        )
