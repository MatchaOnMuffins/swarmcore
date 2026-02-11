from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from swarmcore import Agent, Swarm, chain
from swarmcore.context import SharedContext
from swarmcore.exceptions import AgentError
from tests.conftest import make_mock_response


# ---------------------------------------------------------------------------
# Agent-level timeout / max_retries forwarding
# ---------------------------------------------------------------------------


async def test_agent_timeout_forwarded(mock_llm: AsyncMock):
    """Agent(timeout=30.0) should forward timeout=30.0 to litellm.acompletion."""
    agent = Agent(name="a", instructions="test", timeout=30.0)
    await agent.run("test task", SharedContext())

    mock_llm.assert_called_once()
    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs["timeout"] == 30.0


async def test_agent_max_retries_forwarded(mock_llm: AsyncMock):
    """Agent(max_retries=3) should forward max_retries=3 to litellm.acompletion."""
    agent = Agent(name="a", instructions="test", max_retries=3)
    await agent.run("test task", SharedContext())

    mock_llm.assert_called_once()
    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs["max_retries"] == 3


async def test_agent_no_timeout_by_default(mock_llm: AsyncMock):
    """Agent created without timeout should NOT pass timeout to litellm.acompletion."""
    agent = Agent(name="a", instructions="test")
    await agent.run("test task", SharedContext())

    mock_llm.assert_called_once()
    call_kwargs = mock_llm.call_args.kwargs
    assert "timeout" not in call_kwargs


async def test_agent_no_max_retries_by_default(mock_llm: AsyncMock):
    """Agent created without max_retries should NOT pass max_retries to litellm.acompletion."""
    agent = Agent(name="a", instructions="test")
    await agent.run("test task", SharedContext())

    mock_llm.assert_called_once()
    call_kwargs = mock_llm.call_args.kwargs
    assert "max_retries" not in call_kwargs


# ---------------------------------------------------------------------------
# Agent max_turns
# ---------------------------------------------------------------------------


async def test_agent_max_turns_stops_loop(mock_llm: AsyncMock):
    """Agent with max_turns=2 should stop after 2 LLM calls even if tools keep being requested."""

    def dummy_tool(x: str) -> str:
        """A dummy tool.

        x: input value
        """
        return f"result: {x}"

    # Build a tool call that the LLM will "always" return
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "dummy_tool"
    tool_call.function.arguments = '{"x": "hello"}'

    # Mock returns tool calls on every invocation (never a final text response)
    mock_llm.return_value = make_mock_response(content=None, tool_calls=[tool_call])

    agent = Agent(name="a", instructions="test", tools=[dummy_tool], max_turns=2)

    # The agent should either return a result with truncated output or raise AgentError.
    # We accept either behaviour: the key invariant is that at most 2 LLM calls were made.
    try:
        result = await agent.run("test task", SharedContext())
        # If it returned normally, verify the call count via the result
        assert result.llm_call_count == 2
    except AgentError:
        # Raising is also acceptable when max_turns is exhausted
        pass

    # Regardless of return/raise, the mock should have been called exactly 2 times
    assert mock_llm.call_count == 2


async def test_agent_max_turns_none_is_unlimited(mock_llm: AsyncMock):
    """Agent with max_turns=None (default) should run the tool loop to completion normally."""

    def echo_tool(x: str) -> str:
        """Echo the input.

        x: input value
        """
        return f"echo: {x}"

    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "echo_tool"
    tool_call.function.arguments = '{"x": "hi"}'

    # First call returns a tool call, second returns a final text response
    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="Final answer"),
    ]

    agent = Agent(name="a", instructions="test", tools=[echo_tool])
    result = await agent.run("test task", SharedContext())

    assert result.output == "Final answer"
    assert result.llm_call_count == 2
    assert mock_llm.call_count == 2


# ---------------------------------------------------------------------------
# Swarm-level timeout / max_retries defaults
# ---------------------------------------------------------------------------


async def test_swarm_timeout_default(mock_llm: AsyncMock):
    """Swarm(timeout=30.0) should propagate timeout=30.0 to litellm.acompletion."""
    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(flow=chain(a), timeout=30.0)

    await swarm.run("Test task")

    mock_llm.assert_called_once()
    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs["timeout"] == 30.0


async def test_swarm_max_retries_default(mock_llm: AsyncMock):
    """Swarm(max_retries=5) should propagate max_retries=5 to litellm.acompletion."""
    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(flow=chain(a), max_retries=5)

    await swarm.run("Test task")

    mock_llm.assert_called_once()
    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs["max_retries"] == 5


async def test_agent_overrides_swarm_defaults(mock_llm: AsyncMock):
    """Agent-level timeout should override the Swarm-level default."""
    a = Agent(name="a", instructions="Do A.", timeout=10.0)
    swarm = Swarm(flow=chain(a), timeout=30.0)

    await swarm.run("Test task")

    mock_llm.assert_called_once()
    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs["timeout"] == 10.0
