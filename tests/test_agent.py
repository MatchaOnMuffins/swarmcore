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

    agent = Agent(name="lookup_agent", instructions="Look things up.", tools=[async_lookup])
    ctx = SharedContext()

    result = await agent.run("Look up test", ctx)

    assert result.output == "Found the result."


async def test_agent_unknown_tool_raises(mock_llm: AsyncMock):
    tool_call = MagicMock()
    tool_call.id = "call_789"
    tool_call.function.name = "nonexistent_tool"
    tool_call.function.arguments = "{}"

    mock_llm.return_value = make_mock_response(content=None, tool_calls=[tool_call])

    agent = Agent(name="bad_agent", instructions="Do stuff.")
    ctx = SharedContext()

    with pytest.raises(AgentError, match="unknown tool"):
        await agent.run("Do something", ctx)


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
