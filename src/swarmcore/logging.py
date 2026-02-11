from __future__ import annotations

import dataclasses
import logging
from typing import Any

from swarmcore.hooks import Event, EventType, Hooks

_INFO_EVENTS = {
    EventType.SWARM_START,
    EventType.SWARM_END,
    EventType.STEP_START,
    EventType.STEP_END,
    EventType.AGENT_START,
    EventType.AGENT_END,
}

_DEBUG_EVENTS = {
    EventType.LLM_CALL_START,
    EventType.LLM_CALL_END,
    EventType.TOOL_CALL_START,
    EventType.TOOL_CALL_END,
}

_ERROR_EVENTS = {
    EventType.AGENT_ERROR,
}


class LoggingHandler:
    """Hook handler that logs events via Python's ``logging`` module.

    Swarm/agent lifecycle events are logged at ``INFO``, LLM and tool call
    events at ``DEBUG``, and errors at ``ERROR``.  All event data is passed
    via the ``extra`` dict for structured log formatters.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("swarmcore")

    @staticmethod
    def _as_extra(data: Any) -> dict[str, Any]:
        """Convert event data to a dict suitable for logging ``extra``."""
        if isinstance(data, dict):
            return data
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            return dataclasses.asdict(data)
        return {}

    def __call__(self, event: Event) -> None:
        event_name = event.type.value
        extra = self._as_extra(event.data)

        if event.type in _ERROR_EVENTS:
            self._logger.error("[%s] %s", event_name, event.data, extra=extra)
        elif event.type in _DEBUG_EVENTS:
            self._logger.debug("[%s] %s", event_name, event.data, extra=extra)
        else:
            self._logger.info("[%s] %s", event_name, event.data, extra=extra)


def enable_logging(level: int = logging.INFO) -> Hooks:
    """One-liner to enable structured logging for swarmcore.

    Configures ``logging.basicConfig`` and returns a :class:`Hooks` instance
    with a :class:`LoggingHandler` registered for all events.
    """
    logging.basicConfig(level=level)
    hooks = Hooks()
    hooks.on_all(LoggingHandler())
    return hooks
