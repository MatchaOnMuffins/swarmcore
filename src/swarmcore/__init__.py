"""SwarmCore - Coordinate AI agents in a workflow."""

from swarmcore.agent import Agent
from swarmcore.agents import analyst, editor, researcher, summarizer, writer
from swarmcore.console import ConsoleReporter, console_hooks
from swarmcore.context import SharedContext
from swarmcore.context_tools import make_context_tools
from swarmcore.exceptions import AgentError, SwarmError
from swarmcore.flow import Flow, chain, parallel
from swarmcore.hooks import (
    AgentEndData,
    AgentErrorData,
    AgentRetryData,
    AgentStartData,
    Event,
    EventType,
    Hooks,
    LLMCallEndData,
    LLMCallStartData,
    StepEndData,
    StepStartData,
    SwarmEndData,
    SwarmStartData,
    ToolCallEndData,
    ToolCallStartData,
)
from swarmcore.logging import LoggingHandler, enable_logging
from swarmcore.models import AgentResult, LLMCallRecord, SwarmResult, ToolCallRecord
from swarmcore.swarm import Swarm
from swarmcore.tools import search_web

__all__ = [
    "Agent",
    "AgentEndData",
    "analyst",
    "AgentError",
    "AgentErrorData",
    "AgentRetryData",
    "AgentResult",
    "AgentStartData",
    "ConsoleReporter",
    "Event",
    "EventType",
    "Flow",
    "Hooks",
    "LLMCallEndData",
    "LLMCallStartData",
    "LLMCallRecord",
    "LoggingHandler",
    "SharedContext",
    "StepEndData",
    "StepStartData",
    "Swarm",
    "SwarmEndData",
    "SwarmError",
    "SwarmResult",
    "SwarmStartData",
    "ToolCallEndData",
    "ToolCallRecord",
    "ToolCallStartData",
    "chain",
    "console_hooks",
    "editor",
    "enable_logging",
    "make_context_tools",
    "parallel",
    "researcher",
    "search_web",
    "summarizer",
    "writer",
]
