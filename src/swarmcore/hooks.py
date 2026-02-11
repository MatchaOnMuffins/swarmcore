from __future__ import annotations

import asyncio
import dataclasses
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


# ---------------------------------------------------------------------------
# Typed event data classes
# ---------------------------------------------------------------------------


@dataclass
class _EventDataBase:
    """Base for typed event data with dict-like backward compatibility."""

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible `.get()` — delegates to `getattr`."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Dict-compatible `[]` access — raises `KeyError` on miss."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __iter__(self):  # type: ignore[override]
        """Iterate over field names (dict-key protocol for logging `extra=`)."""
        return iter(f.name for f in dataclasses.fields(self))

    def __contains__(self, key: str) -> bool:
        """Support `key in data` checks."""
        return hasattr(self, key) and key in {f.name for f in dataclasses.fields(self)}

    def items(self) -> list[tuple[str, Any]]:
        """Return all fields as `(key, value)` pairs."""
        return list(dataclasses.asdict(self).items())


@dataclass
class SwarmStartData(_EventDataBase):
    task: str
    step_count: int


@dataclass
class SwarmEndData(_EventDataBase):
    duration_seconds: float
    agent_count: int
    total_cost: float = 0.0


@dataclass
class StepStartData(_EventDataBase):
    step_index: int
    agents: list[str]
    parallel: bool


@dataclass
class StepEndData(_EventDataBase):
    step_index: int


@dataclass
class AgentStartData(_EventDataBase):
    agent: str
    task: str


@dataclass
class AgentEndData(_EventDataBase):
    agent: str
    duration_seconds: float
    cost: float = 0.0


@dataclass
class AgentErrorData(_EventDataBase):
    agent: str
    error: str


@dataclass
class LLMCallStartData(_EventDataBase):
    agent: str
    call_index: int


@dataclass
class LLMCallEndData(_EventDataBase):
    agent: str
    call_index: int
    finish_reason: str
    duration_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ToolCallStartData(_EventDataBase):
    agent: str
    tool: str
    arguments: dict[str, Any]


@dataclass
class ToolCallEndData(_EventDataBase):
    agent: str
    tool: str
    duration_seconds: float


EventData = (
    SwarmStartData
    | SwarmEndData
    | StepStartData
    | StepEndData
    | AgentStartData
    | AgentEndData
    | AgentErrorData
    | LLMCallStartData
    | LLMCallEndData
    | ToolCallStartData
    | ToolCallEndData
)


@dataclass
class Event:
    type: EventType
    data: EventData | dict[str, Any] = field(default_factory=dict)


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
