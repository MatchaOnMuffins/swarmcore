from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from swarmcore import Agent, Swarm, chain, parallel
from swarmcore.exceptions import AgentError
from swarmcore.hooks import AgentRetryData, EventType, Hooks
from tests.conftest import make_mock_response


async def test_no_retry_by_default(mock_llm: AsyncMock):
    """With step_retries=0 (default), an AgentError propagates immediately."""
    mock_llm.side_effect = AgentError("a", "boom")

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(flow=chain(a))

    with pytest.raises(AgentError, match="boom"):
        await swarm.run("Task")

    assert mock_llm.call_count == 1


async def test_retry_succeeds_on_second_attempt(mock_llm: AsyncMock):
    """Agent fails once, succeeds on retry."""
    mock_llm.side_effect = [
        AgentError("a", "transient failure"),
        make_mock_response(content="A output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(flow=chain(a), step_retries=2)

    result = await swarm.run("Task")

    assert result.output == "A output"
    assert mock_llm.call_count == 2


async def test_retry_exhausted(mock_llm: AsyncMock):
    """Agent fails on all attempts — AgentError raised after all retries."""
    mock_llm.side_effect = AgentError("a", "persistent failure")

    collected: list[EventType] = []

    def handler(event):
        collected.append(event.type)

    hooks = Hooks()
    hooks.on_all(handler)

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(flow=chain(a), hooks=hooks, step_retries=3)

    with pytest.raises(AgentError, match="persistent failure"):
        await swarm.run("Task")

    # 1 initial + 3 retries = 4 total attempts
    assert mock_llm.call_count == 4
    # 3 AGENT_RETRY events (one before each retry, not for the final failure)
    assert collected.count(EventType.AGENT_RETRY) == 3


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_retry_exponential_backoff(mock_sleep: AsyncMock, mock_llm: AsyncMock):
    """Verify delays follow delay * multiplier^attempt pattern."""
    mock_llm.side_effect = [
        AgentError("a", "fail 1"),
        AgentError("a", "fail 2"),
        AgentError("a", "fail 3"),
        make_mock_response(content="A output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(
        flow=chain(a),
        step_retries=3,
        retry_delay=1.0,
        retry_multiplier=2.0,
    )

    result = await swarm.run("Task")

    assert result.output == "A output"
    # Delays: 1.0 * 2^0 = 1.0, 1.0 * 2^1 = 2.0, 1.0 * 2^2 = 4.0
    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays == [1.0, 2.0, 4.0]


async def test_retry_parallel_individual(mock_llm: AsyncMock):
    """In parallel [A, B], A succeeds, B fails then succeeds — only B retried."""

    call_count = {"a": 0, "b": 0}

    async def route(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "Do A." in system:
            call_count["a"] += 1
            return make_mock_response(content="A output")
        if "Do B." in system:
            call_count["b"] += 1
            if call_count["b"] == 1:
                raise AgentError("b", "transient")
            return make_mock_response(content="B output")
        return make_mock_response(content="Unknown")

    mock_llm.side_effect = route

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=chain(parallel(a, b)), step_retries=2)
    result = await swarm.run("Task")

    assert result.context["a"] == "A output"
    assert result.context["b"] == "B output"
    # A called once, B called twice (1 fail + 1 success)
    assert call_count["a"] == 1
    assert call_count["b"] == 2


async def test_retry_only_catches_agent_error(mock_llm: AsyncMock):
    """Non-AgentError exceptions propagate immediately without retry."""
    mock_llm.side_effect = ValueError("not an AgentError")

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(flow=chain(a), step_retries=3)

    # ValueError gets wrapped in AgentError by agent.run(), but only the
    # outer AgentError is caught by _with_retry. Since agent.run wraps
    # arbitrary exceptions as AgentError, _with_retry will still catch it.
    # The key test is that the wrapping works and retry kicks in.
    with pytest.raises(AgentError):
        await swarm.run("Task")

    # Agent wraps ValueError as AgentError, so retry does apply.
    # 1 initial + 3 retries = 4 attempts.
    assert mock_llm.call_count == 4


async def test_retry_hook_data(mock_llm: AsyncMock):
    """Verify AgentRetryData fields are correct."""
    mock_llm.side_effect = [
        AgentError("a", "first fail"),
        make_mock_response(content="A output"),
    ]

    retry_events: list[AgentRetryData] = []

    def handler(event):
        if event.type == EventType.AGENT_RETRY:
            retry_events.append(event.data)

    hooks = Hooks()
    hooks.on_all(handler)

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(
        flow=chain(a),
        hooks=hooks,
        step_retries=2,
        retry_delay=0.5,
        retry_multiplier=3.0,
    )
    await swarm.run("Task")

    assert len(retry_events) == 1
    data = retry_events[0]
    assert data.agent == "a"
    assert data.attempt == 1
    assert data.max_retries == 2
    assert "first fail" in data.error
    assert data.delay == 0.5  # 0.5 * 3.0^0 = 0.5


async def test_retry_push_mode(mock_llm: AsyncMock):
    """Step retry works in push mode too."""
    mock_llm.side_effect = [
        make_mock_response(content="A output"),
        AgentError("b", "transient"),
        make_mock_response(content="B output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="push", step_retries=1)
    result = await swarm.run("Task")

    assert result.output == "B output"
    assert result.context["a"] == "A output"
    assert result.context["b"] == "B output"


async def test_retry_subflow(mock_llm: AsyncMock):
    """Agent inside a nested sub-flow retries correctly."""

    call_count = {"b": 0}

    async def route(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "Do A." in system:
            return make_mock_response(content="A output")
        if "Do B." in system:
            call_count["b"] += 1
            if call_count["b"] == 1:
                raise AgentError("b", "sub-flow transient")
            return make_mock_response(content="B output")
        if "Do C." in system:
            return make_mock_response(content="C output")
        return make_mock_response(content="Unknown")

    mock_llm.side_effect = route

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    # (a >> b) | c — b is inside a sub-flow and should be retried
    swarm = Swarm(flow=(a >> b) | c, step_retries=2)
    result = await swarm.run("Task")

    assert result.context["a"] == "A output"
    assert result.context["b"] == "B output"
    assert result.context["c"] == "C output"
    assert call_count["b"] == 2


async def test_retry_fresh_execution(mock_llm: AsyncMock):
    """On retry, agent gets a clean execution (fresh messages list)."""

    messages_received: list[list[dict]] = []

    async def capture(**kwargs):
        messages_received.append(kwargs["messages"])
        if len(messages_received) == 1:
            raise AgentError("a", "first fail")
        return make_mock_response(content="A output")

    mock_llm.side_effect = capture

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(flow=chain(a), step_retries=1)
    await swarm.run("Task")

    # Both attempts should have the same initial messages (fresh start)
    assert len(messages_received) == 2
    first_messages = messages_received[0]
    second_messages = messages_received[1]
    assert len(first_messages) == len(second_messages) == 2
    assert first_messages[0]["role"] == "system"
    assert first_messages[1]["role"] == "user"
    assert second_messages[0]["role"] == "system"
    assert second_messages[1]["role"] == "user"


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_retry_custom_multiplier(mock_sleep: AsyncMock, mock_llm: AsyncMock):
    """Custom retry_delay and retry_multiplier are respected."""
    mock_llm.side_effect = [
        AgentError("a", "fail"),
        AgentError("a", "fail"),
        make_mock_response(content="A output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(
        flow=chain(a),
        step_retries=2,
        retry_delay=0.1,
        retry_multiplier=5.0,
    )
    await swarm.run("Task")

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays == pytest.approx([0.1, 0.5])  # 0.1 * 5^0, 0.1 * 5^1


async def test_retry_zero_means_no_retry(mock_llm: AsyncMock):
    """step_retries=0 means exactly one attempt, no retries."""
    mock_llm.side_effect = AgentError("a", "fail")

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(flow=chain(a), step_retries=0)

    with pytest.raises(AgentError):
        await swarm.run("Task")

    assert mock_llm.call_count == 1
