from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from swarmcore.agent import Agent, _function_to_tool_schema
from swarmcore.context import SharedContext
from swarmcore.exceptions import AgentError
from tests.conftest import make_mock_response


async def test_agent_basic_run(mock_llm: AsyncMock):
    agent = Agent(name="test", instructions="Be helpful.", model="openai/gpt-4o")
    ctx = SharedContext()

    result = await agent.run("Hello", ctx)

    assert result.agent_name == "test"
    assert result.output == "Mock response"
    assert result.model == "openai/gpt-4o"
    assert result.input_task == "Hello"
    assert result.duration_seconds >= 0
    assert result.token_usage.prompt_tokens == 10
    assert result.token_usage.completion_tokens == 20
    assert result.token_usage.total_tokens == 30

    # New observability fields
    assert result.llm_call_count == 1
    assert result.tool_call_count == 0
    assert len(result.llm_calls) == 1
    assert len(result.tool_calls) == 0
    assert result.llm_calls[0].call_index == 0
    assert result.llm_calls[0].finish_reason == "stop"
    assert result.llm_calls[0].token_usage.prompt_tokens == 10
    assert result.llm_calls[0].tool_calls_requested == []

    mock_llm.assert_called_once()
    call_kwargs = mock_llm.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


async def test_agent_context_injection(mock_llm: AsyncMock):
    agent = Agent(name="writer", instructions="Write a report.")
    ctx = SharedContext()
    ctx.set("researcher", "AI is trending in 2025.")

    await agent.run("Write about AI", ctx)

    call_kwargs = mock_llm.call_args
    system_msg = call_kwargs.kwargs["messages"][0]["content"]
    assert "Context from prior agents" in system_msg
    assert "researcher" in system_msg
    assert "AI is trending in 2025." in system_msg


async def test_agent_empty_context(mock_llm: AsyncMock):
    agent = Agent(name="first", instructions="Do research.")
    ctx = SharedContext()

    await agent.run("Research AI", ctx)

    call_kwargs = mock_llm.call_args
    system_msg = call_kwargs.kwargs["messages"][0]["content"]
    assert "Context from prior agents" not in system_msg


async def test_agent_tool_calling(mock_llm: AsyncMock):
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Sunny in {location}"

    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "get_weather"
    tool_call.function.arguments = '{"location": "Paris"}'

    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="The weather in Paris is sunny."),
    ]

    agent = Agent(
        name="weather_agent",
        instructions="Help with weather.",
        tools=[get_weather],
    )
    ctx = SharedContext()

    result = await agent.run("What's the weather in Paris?", ctx)

    assert result.output == "The weather in Paris is sunny."
    assert mock_llm.call_count == 2
    # Token usage accumulated from both calls
    assert result.token_usage.prompt_tokens == 20
    assert result.token_usage.completion_tokens == 40
    assert result.token_usage.total_tokens == 60

    # New observability fields
    assert result.llm_call_count == 2
    assert result.tool_call_count == 1
    assert len(result.llm_calls) == 2
    assert len(result.tool_calls) == 1

    # First LLM call requested a tool
    assert result.llm_calls[0].tool_calls_requested == ["get_weather"]
    # Second LLM call had no tool calls
    assert result.llm_calls[1].tool_calls_requested == []

    # Tool call record
    tc = result.tool_calls[0]
    assert tc.tool_name == "get_weather"
    assert tc.arguments == {"location": "Paris"}
    assert tc.result == "Sunny in Paris"
    assert tc.duration_seconds >= 0


async def test_agent_async_tool(mock_llm: AsyncMock):
    async def async_lookup(query: str) -> str:
        """Look up information."""
        return f"Result for {query}"

    tool_call = MagicMock()
    tool_call.id = "call_456"
    tool_call.function.name = "async_lookup"
    tool_call.function.arguments = '{"query": "test"}'

    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="Found the result."),
    ]

    agent = Agent(
        name="lookup_agent", instructions="Look things up.", tools=[async_lookup]
    )
    ctx = SharedContext()

    result = await agent.run("Look up test", ctx)

    assert result.output == "Found the result."
    assert result.tool_call_count == 1
    assert result.tool_calls[0].tool_name == "async_lookup"


async def test_agent_unknown_tool_returns_error_to_llm(mock_llm: AsyncMock):
    """Unknown tool calls send an error message back to the LLM instead of raising."""
    tool_call = MagicMock()
    tool_call.id = "call_789"
    tool_call.function.name = "nonexistent_tool"
    tool_call.function.arguments = "{}"

    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="Sorry, I couldn't find that tool."),
    ]

    agent = Agent(name="bad_agent", instructions="Do stuff.")
    ctx = SharedContext()

    result = await agent.run("Do something", ctx)

    assert result.output == "Sorry, I couldn't find that tool."
    assert result.tool_call_count == 1
    assert "Error: unknown tool" in result.tool_calls[0].result


async def test_agent_llm_error_raises(mock_llm: AsyncMock):
    mock_llm.side_effect = RuntimeError("API connection failed")

    agent = Agent(name="failing_agent", instructions="Try something.")
    ctx = SharedContext()

    with pytest.raises(AgentError, match="API connection failed"):
        await agent.run("Do something", ctx)


def test_function_to_tool_schema():
    def search(query: str, max_results: int = 10) -> str:
        """Search for information.

        query: The search query
        max_results: Maximum number of results
        """
        return "results"

    schema = _function_to_tool_schema(search)

    assert schema["type"] == "function"
    func = schema["function"]
    assert func["name"] == "search"
    assert func["description"] == "Search for information."
    params = func["parameters"]
    assert params["properties"]["query"]["type"] == "string"
    assert params["properties"]["max_results"]["type"] == "integer"
    assert "query" in params["required"]
    assert "max_results" not in params["required"]


# --- Tiered context tests ---


async def test_structured_output_instruction_injected(mock_llm: AsyncMock):
    agent = Agent(name="test", instructions="Be helpful.")
    ctx = SharedContext()

    await agent.run("Hello", ctx, structured_output=True)

    call_kwargs = mock_llm.call_args
    system_msg = call_kwargs.kwargs["messages"][0]["content"]
    assert "<summary>" in system_msg
    assert "Downstream agents" in system_msg


async def test_structured_output_not_injected_by_default(mock_llm: AsyncMock):
    agent = Agent(name="test", instructions="Be helpful.")
    ctx = SharedContext()

    await agent.run("Hello", ctx)

    call_kwargs = mock_llm.call_args
    system_msg = call_kwargs.kwargs["messages"][0]["content"]
    assert "<summary>" not in system_msg


async def test_expand_param_forwarded_to_context(mock_llm: AsyncMock):
    agent = Agent(name="writer", instructions="Write.")
    ctx = SharedContext()
    ctx.set("a", "Full A output", summary="Summary A")
    ctx.set("b", "Full B output", summary="Summary B")

    await agent.run("Write", ctx, expand={"b"})

    call_kwargs = mock_llm.call_args
    system_msg = call_kwargs.kwargs["messages"][0]["content"]
    assert "Summary A" in system_msg
    assert "Full A output" not in system_msg
    assert "Full B output" in system_msg


async def test_extra_tools_available_during_run(mock_llm: AsyncMock):
    """Extra tools passed to run() should be callable by the LLM."""

    def lookup(query: str) -> str:
        """Look up information.

        query: The search query
        """
        return f"Found: {query}"

    tool_call = MagicMock()
    tool_call.id = "call_extra"
    tool_call.function.name = "lookup"
    tool_call.function.arguments = '{"query": "test"}'

    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="Done."),
    ]

    agent = Agent(name="test", instructions="Hi.")
    ctx = SharedContext()

    result = await agent.run("Go", ctx, extra_tools=[lookup])

    assert result.output == "Done."
    assert result.tool_call_count == 1
    assert result.tool_calls[0].tool_name == "lookup"
    assert result.tool_calls[0].result == "Found: test"


async def test_extra_tools_do_not_persist(mock_llm: AsyncMock):
    """Extra tools from one run() should not leak into the next."""

    def ephemeral(x: str) -> str:
        """Temp tool."""
        return x

    agent = Agent(name="test", instructions="Hi.")
    ctx = SharedContext()

    # First run with extra tool
    await agent.run("Go", ctx, extra_tools=[ephemeral])

    # Second run without — LLM tries to call it, should get error result (not raise)
    tool_call = MagicMock()
    tool_call.id = "call_leak"
    tool_call.function.name = "ephemeral"
    tool_call.function.arguments = '{"x": "hi"}'

    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="Tool not available."),
    ]

    result = await agent.run("Go again", ctx)
    assert result.output == "Tool not available."
    assert "Error: unknown tool" in result.tool_calls[0].result


async def test_agent_tool_invalid_json_args(mock_llm: AsyncMock):
    """Malformed JSON in tool arguments sends an error back to the LLM."""

    def my_tool(x: str) -> str:
        """A tool."""
        return x

    tool_call = MagicMock()
    tool_call.id = "call_bad_json"
    tool_call.function.name = "my_tool"
    tool_call.function.arguments = "NOT VALID JSON"

    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="I'll try a different approach."),
    ]

    agent = Agent(name="test", instructions="Hi.", tools=[my_tool])
    ctx = SharedContext()

    result = await agent.run("Go", ctx)

    assert result.output == "I'll try a different approach."
    assert result.tool_call_count == 1
    assert "Error: invalid arguments JSON" in result.tool_calls[0].result


async def test_agent_tool_execution_error(mock_llm: AsyncMock):
    """Tool that raises an exception sends the error back to the LLM."""

    def failing_tool(x: str) -> str:
        """A tool that fails."""
        raise ValueError("something broke")

    tool_call = MagicMock()
    tool_call.id = "call_fail"
    tool_call.function.name = "failing_tool"
    tool_call.function.arguments = '{"x": "test"}'

    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="The tool failed, but I can handle it."),
    ]

    agent = Agent(name="test", instructions="Hi.", tools=[failing_tool])
    ctx = SharedContext()

    result = await agent.run("Go", ctx)

    assert result.output == "The tool failed, but I can handle it."
    assert result.tool_call_count == 1
    assert (
        "Error: tool 'failing_tool' failed: something broke"
        in result.tool_calls[0].result
    )


async def test_agent_async_tool_execution_error(mock_llm: AsyncMock):
    """Async tool that raises an exception sends the error back to the LLM."""

    async def async_failing(x: str) -> str:
        """An async tool that fails."""
        raise RuntimeError("async failure")

    tool_call = MagicMock()
    tool_call.id = "call_async_fail"
    tool_call.function.name = "async_failing"
    tool_call.function.arguments = '{"x": "test"}'

    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="Recovered from async failure."),
    ]

    agent = Agent(name="test", instructions="Hi.", tools=[async_failing])
    ctx = SharedContext()

    result = await agent.run("Go", ctx)

    assert result.output == "Recovered from async failure."
    assert (
        "Error: tool 'async_failing' failed: async failure"
        in result.tool_calls[0].result
    )
