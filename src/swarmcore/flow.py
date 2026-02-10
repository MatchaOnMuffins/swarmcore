from __future__ import annotations

from typing import TYPE_CHECKING

from swarmcore.exceptions import SwarmError

if TYPE_CHECKING:
    from swarmcore.agent import Agent


class _ParallelGroup:
    """Marker returned by ``parallel()`` and consumed by ``chain()``."""

    def __init__(self, items: list[Agent | Flow]) -> None:
        self.items = items


def _or_items(value: Agent | Flow) -> list[Agent | Flow]:
    """Normalize a value for use in the ``|`` operator.

    - Single Agent → ``[agent]``
    - Single-agent Flow → ``[agent]``  (unwrap)
    - Single parallel-group Flow → the group's items  (merge)
    - Multi-step Flow → ``[flow]``  (keep as sub-flow)
    """
    from swarmcore.agent import Agent

    if isinstance(value, Agent):
        return [value]
    if isinstance(value, Flow):
        if len(value._steps) == 1:
            step = value._steps[0]
            if isinstance(step, list):
                # Single parallel-group: merge its items
                return list(step)
            # Single-agent Flow: unwrap
            return [step]
        # Multi-step Flow: keep as sub-flow
        return [value]
    return []


class Flow:
    """Immutable execution plan holding a sequence of steps.

    Each step is either a single :class:`Agent` (sequential) or a
    ``list[Agent | Flow]`` (parallel).  A ``Flow`` inside a parallel
    group runs its own steps sequentially within that concurrent group.
    """

    def __init__(self, steps: list[Agent | list[Agent | Flow]]) -> None:
        self._steps: list[Agent | list[Agent | Flow]] = list(steps)

    @property
    def steps(self) -> list[Agent | list[Agent | Flow]]:
        return list(self._steps)

    @property
    def agents(self) -> list[Agent]:
        """All unique agents in step order, recursing into sub-flows."""
        seen: set[str] = set()
        result: list[Agent] = []
        for step in self._steps:
            if isinstance(step, list):
                for item in step:
                    if isinstance(item, Flow):
                        for agent in item.agents:
                            if agent.name not in seen:
                                seen.add(agent.name)
                                result.append(agent)
                    else:
                        if item.name not in seen:
                            seen.add(item.name)
                            result.append(item)
            else:
                if step.name not in seen:
                    seen.add(step.name)
                    result.append(step)
        return result

    def __rshift__(self, other: Agent | Flow) -> Flow:
        from swarmcore.agent import Agent

        if isinstance(other, Agent):
            return Flow(self._steps + [other])
        if isinstance(other, Flow):
            return Flow(self._steps + other._steps)
        return NotImplemented

    def __or__(self, other: Agent | Flow) -> Flow:
        # Normalize other into a list of items to put in the parallel group
        other_items: list[Agent | Flow] = _or_items(other)
        if not other_items:
            return NotImplemented  # type: ignore[return-value]

        # Normalize self into a list of items already in the parallel group
        self_items: list[Agent | Flow] = _or_items(self)

        # If self was already a parallel group (possibly with prefix steps),
        # extend it.  Otherwise combine self_items + other_items.
        if self._steps and isinstance(self._steps[-1], list):
            # self ends with a parallel group — extend it
            new_steps: list[Agent | list[Agent | Flow]] = list(self._steps)
            new_steps[-1] = list(self._steps[-1]) + other_items
            return Flow(new_steps)

        return Flow([self_items + other_items])

    def __repr__(self) -> str:
        parts: list[str] = []
        for step in self._steps:
            if isinstance(step, list):
                item_strs: list[str] = []
                for item in step:
                    if isinstance(item, Flow):
                        item_strs.append(f"({repr(item)[5:-1]})")
                    else:
                        item_strs.append(item.name)
                parts.append(f"[{', '.join(item_strs)}]")
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

    steps: list[Agent | list[Agent | Flow]] = []
    for item in items:
        if isinstance(item, _ParallelGroup):
            steps.append(item.items)
        elif isinstance(item, Agent):
            steps.append(item)
        else:
            raise SwarmError(
                f"chain() accepts Agent or parallel() groups, got {type(item).__name__}"
            )
    return Flow(steps)


def parallel(*items: Agent | Flow) -> _ParallelGroup:
    """Group agents or sub-flows for concurrent execution within a :func:`chain`.

    Requires at least two items.  Each item may be an :class:`Agent`
    (run individually) or a :class:`Flow` (run its steps sequentially
    within the concurrent group).
    """
    if len(items) < 2:
        raise SwarmError("parallel() requires at least 2 agents")
    return _ParallelGroup(list(items))
