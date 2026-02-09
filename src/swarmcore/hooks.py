from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("swarmcore")

# Handler type: sync or async callable accepting an Event
Handler = Callable[["Event"], Any]


class EventType(Enum):
    SWARM_START = "swarm_start"
    SWARM_END = "swarm_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)


class Hooks:
    """Lightweight callback system for execution events."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = {}
        self._global_handlers: list[Handler] = []

    def on(self, event_type: EventType, handler: Handler) -> None:
        """Register a handler for a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def on_all(self, handler: Handler) -> None:
        """Register a handler that receives all events."""
        self._global_handlers.append(handler)

    @property
    def is_active(self) -> bool:
        """True when at least one handler is registered."""
        return bool(self._global_handlers) or any(self._handlers.values())

    async def emit(self, event: Event) -> None:
        """Dispatch an event to registered handlers.

        Supports both sync and async handlers. Handler exceptions are
        logged and swallowed so they never break the execution flow.
        """
        handlers = list(self._global_handlers)
        handlers.extend(self._handlers.get(event.type, []))

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Hook handler %r failed for event %s", handler, event.type.value
                )
