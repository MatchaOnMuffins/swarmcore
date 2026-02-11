from __future__ import annotations

import pytest

from swarmcore.hooks import (
    AgentEndData,
    AgentErrorData,
    AgentStartData,
    Event,
    EventType,
    LLMCallEndData,
    LLMCallStartData,
    StepEndData,
    StepStartData,
    SwarmEndData,
    SwarmStartData,
    ToolCallEndData,
    ToolCallStartData,
)


# ---------------------------------------------------------------------------
# Typed attribute access
# ---------------------------------------------------------------------------


def test_agent_start_typed_access() -> None:
    data = AgentStartData(agent="a", task="t")
    assert data.agent == "a"
    assert data.task == "t"


# ---------------------------------------------------------------------------
# Dict-compat .get() access
# ---------------------------------------------------------------------------


def test_get_access() -> None:
    data = AgentStartData(agent="a", task="t")
    assert data.get("agent") == "a"
    assert data.get("task") == "t"


def test_get_default_for_missing_key() -> None:
    data = AgentStartData(agent="a", task="t")
    assert data.get("nonexistent") is None
    assert data.get("nonexistent", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Dict-compat [] access
# ---------------------------------------------------------------------------


def test_bracket_access() -> None:
    data = AgentStartData(agent="a", task="t")
    assert data["agent"] == "a"
    assert data["task"] == "t"


def test_bracket_raises_key_error_for_missing() -> None:
    data = AgentStartData(agent="a", task="t")
    with pytest.raises(KeyError):
        data["nonexistent"]


# ---------------------------------------------------------------------------
# .items()
# ---------------------------------------------------------------------------


def test_items_returns_all_fields() -> None:
    data = AgentStartData(agent="a", task="t")
    items = dict(data.items())
    assert items == {"agent": "a", "task": "t"}


# ---------------------------------------------------------------------------
# All 11 data classes can be constructed and accessed
# ---------------------------------------------------------------------------


def test_swarm_start_data() -> None:
    data = SwarmStartData(task="do stuff", step_count=3)
    assert data.task == "do stuff"
    assert data.step_count == 3
    assert data.get("task") == "do stuff"
    assert data["step_count"] == 3
    assert dict(data.items()) == {"task": "do stuff", "step_count": 3}


def test_swarm_end_data() -> None:
    data = SwarmEndData(duration_seconds=1.5, agent_count=2)
    assert data.duration_seconds == 1.5
    assert data.agent_count == 2
    assert data.get("duration_seconds") == 1.5
    assert data["agent_count"] == 2


def test_step_start_data() -> None:
    data = StepStartData(step_index=0, agents=["a", "b"], parallel=True)
    assert data.step_index == 0
    assert data.agents == ["a", "b"]
    assert data.parallel is True
    assert data.get("parallel") is True
    assert data["agents"] == ["a", "b"]


def test_step_end_data() -> None:
    data = StepEndData(step_index=1)
    assert data.step_index == 1
    assert data.get("step_index") == 1
    assert data["step_index"] == 1


def test_agent_start_data() -> None:
    data = AgentStartData(agent="writer", task="write poem")
    assert data.agent == "writer"
    assert data.task == "write poem"
    assert data.get("agent") == "writer"
    assert data["task"] == "write poem"


def test_agent_end_data() -> None:
    data = AgentEndData(agent="writer", duration_seconds=2.3)
    assert data.agent == "writer"
    assert data.duration_seconds == 2.3
    assert data.get("duration_seconds") == 2.3
    assert data["agent"] == "writer"


def test_agent_error_data() -> None:
    data = AgentErrorData(agent="writer", error="timeout")
    assert data.agent == "writer"
    assert data.error == "timeout"
    assert data.get("error") == "timeout"
    assert data["agent"] == "writer"


def test_llm_call_start_data() -> None:
    data = LLMCallStartData(agent="writer", call_index=0)
    assert data.agent == "writer"
    assert data.call_index == 0
    assert data.get("call_index") == 0
    assert data["agent"] == "writer"


def test_llm_call_end_data() -> None:
    data = LLMCallEndData(
        agent="writer",
        call_index=1,
        finish_reason="stop",
        duration_seconds=0.5,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    assert data.agent == "writer"
    assert data.call_index == 1
    assert data.finish_reason == "stop"
    assert data.duration_seconds == 0.5
    assert data.prompt_tokens == 100
    assert data.completion_tokens == 50
    assert data.total_tokens == 150
    assert data.get("finish_reason") == "stop"
    assert data["total_tokens"] == 150
    items = dict(data.items())
    assert len(items) == 7
    assert items["prompt_tokens"] == 100


def test_tool_call_start_data() -> None:
    data = ToolCallStartData(agent="writer", tool="search", arguments={"q": "hello"})
    assert data.agent == "writer"
    assert data.tool == "search"
    assert data.arguments == {"q": "hello"}
    assert data.get("tool") == "search"
    assert data["arguments"] == {"q": "hello"}


def test_tool_call_end_data() -> None:
    data = ToolCallEndData(agent="writer", tool="search", duration_seconds=0.1)
    assert data.agent == "writer"
    assert data.tool == "search"
    assert data.duration_seconds == 0.1
    assert data.get("duration_seconds") == 0.1
    assert data["tool"] == "search"


# ---------------------------------------------------------------------------
# Event with typed data (backward compat with dict still works)
# ---------------------------------------------------------------------------


def test_event_with_typed_data() -> None:
    data = AgentStartData(agent="a", task="t")
    event = Event(type=EventType.AGENT_START, data=data)
    assert event.data.get("agent") == "a"
    assert event.data["task"] == "t"


def test_event_with_dict_data() -> None:
    event = Event(type=EventType.AGENT_START, data={"agent": "a", "task": "t"})
    assert event.data["agent"] == "a"
    assert event.data.get("task") == "t"


def test_event_default_data_is_empty_dict() -> None:
    event = Event(type=EventType.SWARM_START)
    assert event.data == {}
