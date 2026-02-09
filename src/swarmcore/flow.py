from __future__ import annotations

from typing import TYPE_CHECKING

from swarmcore.exceptions import SwarmError

if TYPE_CHECKING:
    from swarmcore.agent import Agent


class _ParallelGroup:
    """Marker returned by ``parallel()`` and consumed by ``chain()``."""

    def __init__(self, agents: list[Agent]) -> None:
        self.agents = agents


class Flow:
    """Immutable execution plan holding a sequence of steps.

    Each step is either a single :class:`Agent` (sequential) or a
    ``list[Agent]`` (parallel).
    """

    def __init__(self, steps: list[Agent | list[Agent]]) -> None:
        self._steps: list[Agent | list[Agent]] = list(steps)

    @property
    def steps(self) -> list[Agent | list[Agent]]:
        return list(self._steps)

    @property
    def agents(self) -> list[Agent]:
        """All unique agents in step order."""
        seen: set[str] = set()
        result: list[Agent] = []
        for step in self._steps:
            items = step if isinstance(step, list) else [step]
            for agent in items:
                if agent.name not in seen:
                    seen.add(agent.name)
                    result.append(agent)
        return result

    def __rshift__(self, other: Agent | Flow) -> Flow:
        from swarmcore.agent import Agent

        if isinstance(other, Agent):
            return Flow(self._steps + [other])
        if isinstance(other, Flow):
            return Flow(self._steps + other._steps)
        return NotImplemented

    def __or__(self, other: Agent | Flow) -> Flow:
        from swarmcore.agent import Agent

        if isinstance(other, Agent):
            if self._steps and isinstance(self._steps[-1], list):
                new_steps = list(self._steps)
                new_steps[-1] = list(self._steps[-1]) + [other]
                return Flow(new_steps)
            if self._steps and not isinstance(self._steps[-1], list):
                new_steps = list(self._steps[:-1])
                new_steps.append([self._steps[-1], other])
                return Flow(new_steps)
            return Flow([[other]])
        if isinstance(other, Flow):
            # Merge all agents from other into a parallel group with last step
            all_other_agents: list[Agent] = []
            for step in other._steps:
                if isinstance(step, list):
                    all_other_agents.extend(step)
                else:
                    all_other_agents.append(step)
            if self._steps and isinstance(self._steps[-1], list):
                new_steps = list(self._steps)
                new_steps[-1] = list(self._steps[-1]) + all_other_agents
                return Flow(new_steps)
            if self._steps and not isinstance(self._steps[-1], list):
                new_steps = list(self._steps[:-1])
                new_steps.append([self._steps[-1]] + all_other_agents)
                return Flow(new_steps)
            return Flow([all_other_agents])
        return NotImplemented

    def __repr__(self) -> str:
        parts: list[str] = []
        for step in self._steps:
            if isinstance(step, list):
                names = ", ".join(a.name for a in step)
                parts.append(f"[{names}]")
            else:
                parts.append(step.name)
        return f"Flow({' >> '.join(parts)})"


def chain(*items: Agent | _ParallelGroup) -> Flow:
    """Compose agents into a sequential flow.

    Use :func:`parallel` to create parallel steps within the chain::

        chain(researcher, parallel(analyst, writer), editor)
    """
    from swarmcore.agent import Agent

    if not items:
        raise SwarmError("chain() requires at least one agent")

    steps: list[Agent | list[Agent]] = []
    for item in items:
        if isinstance(item, _ParallelGroup):
            steps.append(item.agents)
        elif isinstance(item, Agent):
            steps.append(item)
        else:
            raise SwarmError(
                f"chain() accepts Agent or parallel() groups, got {type(item).__name__}"
            )
    return Flow(steps)


def parallel(*agents: Agent) -> _ParallelGroup:
    """Group agents for concurrent execution within a :func:`chain`.

    Requires at least two agents.
    """
    if len(agents) < 2:
        raise SwarmError("parallel() requires at least 2 agents")
    return _ParallelGroup(list(agents))
