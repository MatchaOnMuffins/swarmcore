from __future__ import annotations

import asyncio

from swarmcore.agent import Agent
from swarmcore.context import SharedContext
from swarmcore.exceptions import SwarmError
from swarmcore.models import AgentResult, SwarmResult


def parse_flow(flow_str: str) -> list[str | list[str]]:
    """Parse a flow string into a list of execution steps.

    Examples:
        "a >> b >> c"       -> ["a", "b", "c"]
        "a >> [b, c] >> d"  -> ["a", ["b", "c"], "d"]
    """
    if not flow_str or not flow_str.strip():
        raise SwarmError("Flow string cannot be empty")

    steps: list[str | list[str]] = []
    raw_tokens = flow_str.split(">>")

    for token in raw_tokens:
        token = token.strip()
        if not token:
            raise SwarmError("Flow contains empty step (consecutive '>>' tokens)")

        if token.startswith("[") and token.endswith("]"):
            inner = token[1:-1]
            names = [name.strip() for name in inner.split(",")]
            names = [n for n in names if n]
            if not names:
                raise SwarmError("Flow contains empty parallel group '[]'")
            steps.append(names)
        else:
            if "[" in token or "]" in token:
                raise SwarmError(f"Malformed parallel group in flow: '{token}'")
            steps.append(token)

    return steps


class Swarm:
    def __init__(
        self,
        agents: list[Agent],
        flow: str,
    ) -> None:
        self._agents: dict[str, Agent] = {a.name: a for a in agents}
        self._flow_str = flow
        self._steps = parse_flow(flow)

        for step in self._steps:
            names = step if isinstance(step, list) else [step]
            for name in names:
                if name not in self._agents:
                    raise SwarmError(
                        f"Flow references agent '{name}' but no agent with that name "
                        f"was provided. Available agents: {sorted(self._agents.keys())}"
                    )

    async def run(self, task: str) -> SwarmResult:
        """Execute the swarm workflow on the given task."""
        context = SharedContext()
        history: list[AgentResult] = []

        for step in self._steps:
            if isinstance(step, list):
                coros = [self._agents[name].run(task, context) for name in step]
                results = await asyncio.gather(*coros)

                for result in results:
                    context.set(result.agent_name, result.output)
                    history.append(result)
            else:
                agent = self._agents[step]
                result = await agent.run(task, context)
                context.set(result.agent_name, result.output)
                history.append(result)

        return SwarmResult(
            output=history[-1].output,
            context=context.to_dict(),
            history=history,
        )
