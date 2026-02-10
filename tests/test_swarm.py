from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from swarmcore import Agent, Swarm, SwarmResult, chain, parallel
from swarmcore.hooks import EventType, Hooks
from swarmcore.models import AgentResult
from tests.conftest import make_mock_response


async def test_sequential_flow(mock_llm: AsyncMock):
    mock_llm.side_effect = [
        make_mock_response(content="Research output"),
        make_mock_response(content="Writer output"),
    ]

    researcher = Agent(name="researcher", instructions="Research.")
    writer = Agent(name="writer", instructions="Write.")

    swarm = Swarm(flow=researcher >> writer, context_mode="push")

    result = await swarm.run("Test task")

    assert result.output == "Writer output"
    assert result.context["researcher"] == "Research output"
    assert result.context["writer"] == "Writer output"
    assert len(result.history) == 2
    assert result.history[0].agent_name == "researcher"
    assert result.history[1].agent_name == "writer"


async def test_parallel_flow(mock_llm: AsyncMock):
    mock_llm.side_effect = [
        make_mock_response(content="Agent A output"),
        make_mock_response(content="Agent B output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=chain(parallel(a, b)), context_mode="push")

    result = await swarm.run("Test task")

    assert result.context["a"] == "Agent A output"
    assert result.context["b"] == "Agent B output"
    assert len(result.history) == 2


async def test_mixed_flow(mock_llm: AsyncMock):
    mock_llm.side_effect = [
        make_mock_response(content="Planner output"),
        make_mock_response(content="Researcher output"),
        make_mock_response(content="Critic output"),
        make_mock_response(content="Writer output"),
    ]

    planner = Agent(name="planner", instructions="Plan.")
    researcher = Agent(name="researcher", instructions="Research.")
    critic = Agent(name="critic", instructions="Critique.")
    writer = Agent(name="writer", instructions="Write.")

    swarm = Swarm(
        flow=chain(planner, parallel(researcher, critic), writer),
        context_mode="push",
    )

    result = await swarm.run("Test task")

    assert result.output == "Writer output"
    assert len(result.context) == 4
    assert len(result.history) == 4
    assert result.history[0].agent_name == "planner"
    assert result.history[-1].agent_name == "writer"


async def test_context_passing(mock_llm: AsyncMock):
    mock_llm.side_effect = [
        make_mock_response(content="Research notes here"),
        make_mock_response(content="Summary based on research"),
    ]

    researcher = Agent(name="researcher", instructions="Research.")
    writer = Agent(name="writer", instructions="Write.")

    swarm = Swarm(flow=researcher >> writer, context_mode="push")

    await swarm.run("Test task")

    # The second call (writer) should have context from the first (researcher)
    second_call = mock_llm.call_args_list[1]
    system_msg = second_call.kwargs["messages"][0]["content"]
    assert "researcher" in system_msg
    assert "Research notes here" in system_msg


async def test_result_structure(mock_llm: AsyncMock):
    solo = Agent(name="solo", instructions="Work alone.")

    swarm = Swarm(flow=chain(solo), context_mode="push")

    result = await swarm.run("Do it")

    assert isinstance(result, SwarmResult)
    assert isinstance(result.output, str)
    assert isinstance(result.context, dict)
    assert isinstance(result.history, list)
    assert all(isinstance(r, AgentResult) for r in result.history)


async def test_swarm_duration_and_total_usage(mock_llm: AsyncMock):
    mock_llm.side_effect = [
        make_mock_response(content="A output", prompt_tokens=5, completion_tokens=10),
        make_mock_response(content="B output", prompt_tokens=15, completion_tokens=25),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="push")
    result = await swarm.run("Task")

    assert result.duration_seconds >= 0
    assert result.total_token_usage.prompt_tokens == 20
    assert result.total_token_usage.completion_tokens == 35
    assert result.total_token_usage.total_tokens == 55


async def test_swarm_with_hooks(mock_llm: AsyncMock):
    mock_llm.side_effect = [
        make_mock_response(content="Output A"),
        make_mock_response(content="Output B"),
    ]

    collected: list[EventType] = []

    def handler(event):
        collected.append(event.type)

    hooks = Hooks()
    hooks.on_all(handler)

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, hooks=hooks, context_mode="push")
    await swarm.run("Task")

    assert EventType.SWARM_START in collected
    assert EventType.SWARM_END in collected
    assert EventType.STEP_START in collected
    assert EventType.STEP_END in collected
    assert EventType.AGENT_START in collected
    assert EventType.AGENT_END in collected
    assert EventType.LLM_CALL_START in collected
    assert EventType.LLM_CALL_END in collected


# --- Tiered context tests (push mode) ---


async def test_tiered_context_sequential_chain(mock_llm: AsyncMock):
    """A >> B >> C: C sees A's summary + B's full output."""
    mock_llm.side_effect = [
        make_mock_response(content="<summary>A summary.</summary>\nA detailed output."),
        make_mock_response(content="<summary>B summary.</summary>\nB detailed output."),
        make_mock_response(content="C final output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    swarm = Swarm(flow=a >> b >> c, context_mode="push")
    result = await swarm.run("Task")

    # A's output should have summary stripped
    assert result.history[0].output == "A detailed output."
    assert result.history[0].summary == "A summary."

    # B's output should have summary stripped
    assert result.history[1].output == "B detailed output."
    assert result.history[1].summary == "B summary."

    # C's call: system prompt should contain A's summary and B's full output
    c_call = mock_llm.call_args_list[2]
    c_system = c_call.kwargs["messages"][0]["content"]
    assert "A summary." in c_system
    assert "A detailed output." not in c_system
    assert "B detailed output." in c_system

    # C gracefully degrades (no tags)
    assert result.output == "C final output."
    assert result.history[2].summary == "C final output."


async def test_tiered_context_parallel_flow(mock_llm: AsyncMock):
    """A >> (B | C) >> D: D sees A's summary + B's full + C's full."""
    mock_llm.side_effect = [
        make_mock_response(content="<summary>A sum.</summary>\nA detail."),
        make_mock_response(content="<summary>B sum.</summary>\nB detail."),
        make_mock_response(content="<summary>C sum.</summary>\nC detail."),
        make_mock_response(content="D output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")
    d = Agent(name="d", instructions="Do D.")

    swarm = Swarm(flow=chain(a, parallel(b, c), d), context_mode="push")
    result = await swarm.run("Task")

    # D's system prompt should have A's summary, B's and C's full output
    d_call = mock_llm.call_args_list[3]
    d_system = d_call.kwargs["messages"][0]["content"]
    assert "A sum." in d_system
    assert "A detail." not in d_system
    assert "B detail." in d_system
    assert "C detail." in d_system

    assert result.output == "D output."


async def test_graceful_degradation_no_tags(mock_llm: AsyncMock):
    """When no <summary> tags, output is used as both summary and detail."""
    mock_llm.side_effect = [
        make_mock_response(content="Plain A output"),
        make_mock_response(content="Plain B output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="push")
    result = await swarm.run("Task")

    assert result.history[0].output == "Plain A output"
    assert result.history[0].summary == "Plain A output"
    assert result.output == "Plain B output"


async def test_swarm_result_output_is_clean(mock_llm: AsyncMock):
    """SwarmResult.output should not contain <summary> tags."""
    mock_llm.side_effect = [
        make_mock_response(content="<summary>Final summary.</summary>\nFinal detail."),
    ]

    solo = Agent(name="solo", instructions="Work.")
    swarm = Swarm(flow=chain(solo), context_mode="push")
    result = await swarm.run("Task")

    assert "<summary>" not in result.output
    assert result.output == "Final detail."


async def test_expand_tool_injected_when_summaries_exist(mock_llm: AsyncMock):
    """A >> B >> C: at step C, A is summarized so expand_context tool is available.

    C's LLM calls expand_context("a") to get A's full output.
    """
    tool_call = MagicMock()
    tool_call.id = "call_expand"
    tool_call.function.name = "expand_context"
    tool_call.function.arguments = '{"agent_name": "a"}'

    mock_llm.side_effect = [
        make_mock_response(content="<summary>A sum.</summary>\nA detail."),
        make_mock_response(content="<summary>B sum.</summary>\nB detail."),
        # C first calls the expand tool, then produces final output
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="C output using A's full data."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    swarm = Swarm(flow=a >> b >> c, context_mode="push")
    result = await swarm.run("Task")

    assert result.output == "C output using A's full data."

    # C should have recorded a tool call for expand_context
    c_result = result.history[2]
    assert c_result.tool_call_count == 1
    assert c_result.tool_calls[0].tool_name == "expand_context"
    assert "A detail." in c_result.tool_calls[0].result


async def test_expand_hint_in_system_prompt(mock_llm: AsyncMock):
    """When expand_context tool is available, agent's system prompt should mention it."""
    mock_llm.side_effect = [
        make_mock_response(content="<summary>A sum.</summary>\nA detail."),
        make_mock_response(content="<summary>B sum.</summary>\nB detail."),
        make_mock_response(content="C output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    swarm = Swarm(flow=a >> b >> c, context_mode="push")
    await swarm.run("Task")

    # C has summarized entries (A), so its system prompt should hint about expand_context
    c_call = mock_llm.call_args_list[2]
    c_system = c_call.kwargs["messages"][0]["content"]
    assert "expand_context" in c_system

    # B has no summarized entries (only A at full), so no hint
    b_call = mock_llm.call_args_list[1]
    b_system = b_call.kwargs["messages"][0]["content"]
    assert "expand_context" not in b_system


async def test_expand_tool_not_injected_on_first_step(mock_llm: AsyncMock):
    """First agent has no prior context, so no expand tool should be available.

    The LLM hallucinates expand_context, gets an error, then recovers.
    """
    tool_call = MagicMock()
    tool_call.id = "call_bad"
    tool_call.function.name = "expand_context"
    tool_call.function.arguments = '{"agent_name": "nobody"}'

    mock_llm.side_effect = [
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="A output after recovery."),
    ]

    a = Agent(name="a", instructions="Do A.")
    swarm = Swarm(flow=chain(a), context_mode="push")

    result = await swarm.run("Task")
    assert result.output == "A output after recovery."
    a_result = result.history[0]
    assert a_result.tool_call_count == 1
    assert "Error: unknown tool" in a_result.tool_calls[0].result


async def test_expand_tool_not_injected_when_all_expanded(mock_llm: AsyncMock):
    """A >> B: B already sees A's full output, so no expand tool needed."""
    mock_llm.side_effect = [
        make_mock_response(content="A output."),
        make_mock_response(content="B output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="push")
    await swarm.run("Task")

    # B's call should NOT have expand_context in its tools
    b_call = mock_llm.call_args_list[1]
    tools = b_call.kwargs.get("tools", [])
    tool_names = [t["function"]["name"] for t in tools]
    assert "expand_context" not in tool_names


# --- Pull-mode tests ---


async def test_pull_mode_injects_context_tools(mock_llm: AsyncMock):
    """In pull mode, second agent gets context tools instead of expand_context."""
    mock_llm.side_effect = [
        make_mock_response(content="A output."),
        make_mock_response(content="B output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="pull")
    await swarm.run("Task")

    # B's call should have pull-mode context tools
    b_call = mock_llm.call_args_list[1]
    tools = b_call.kwargs.get("tools", [])
    tool_names = {t["function"]["name"] for t in tools}
    assert "list_context" in tool_names
    assert "get_context" in tool_names
    assert "search_context" in tool_names
    assert "expand_context" not in tool_names


async def test_pull_mode_no_full_context_in_system_prompt(mock_llm: AsyncMock):
    """Pull mode should NOT inject full detail into system prompt — only summary."""
    mock_llm.side_effect = [
        make_mock_response(
            content="<summary>A short summary.</summary>\nDetailed research output with many words."
        ),
        make_mock_response(content="B output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="pull")
    await swarm.run("Task")

    b_call = mock_llm.call_args_list[1]
    b_system = b_call.kwargs["messages"][0]["content"]
    # Should NOT contain the full detail text
    assert "Detailed research output with many words." not in b_system
    # Should contain the summary
    assert "A short summary." in b_system
    # Should contain the agent name as a hint
    assert "a" in b_system
    # Should contain instructions about context tools
    assert "list_context" in b_system or "get_context" in b_system


async def test_pull_mode_includes_lightweight_hint(mock_llm: AsyncMock):
    """Pull mode hint includes agent names and summaries."""
    mock_llm.side_effect = [
        make_mock_response(
            content="<summary>Research summary.</summary>\nFull research."
        ),
        make_mock_response(content="B output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="pull")
    await swarm.run("Task")

    b_call = mock_llm.call_args_list[1]
    b_system = b_call.kwargs["messages"][0]["content"]
    assert "Available context" in b_system
    assert "a" in b_system
    assert "Research summary." in b_system


async def test_pull_mode_first_agent_no_context_tools(mock_llm: AsyncMock):
    """First agent in pull mode should get no context tools."""
    mock_llm.side_effect = [
        make_mock_response(content="A output."),
        make_mock_response(content="B output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="pull")
    await swarm.run("Task")

    # First agent's call should NOT have context tools
    a_call = mock_llm.call_args_list[0]
    tools = a_call.kwargs.get("tools", [])
    if tools:
        tool_names = {t["function"]["name"] for t in tools}
        assert "list_context" not in tool_names
        assert "get_context" not in tool_names
        assert "search_context" not in tool_names


async def test_pull_mode_agent_calls_get_context(mock_llm: AsyncMock):
    """Agent in pull mode can call get_context to retrieve prior output."""
    tool_call = MagicMock()
    tool_call.id = "call_get"
    tool_call.function.name = "get_context"
    tool_call.function.arguments = '{"agent_name": "a"}'

    mock_llm.side_effect = [
        make_mock_response(content="A detailed output."),
        # B first calls get_context, then produces final output
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="B output using A's data."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="pull")
    result = await swarm.run("Task")

    assert result.output == "B output using A's data."

    # B should have recorded a tool call for get_context
    b_result = result.history[1]
    assert b_result.tool_call_count == 1
    assert b_result.tool_calls[0].tool_name == "get_context"
    assert "A detailed output." in b_result.tool_calls[0].result


async def test_pull_mode_summary_parsing_works(mock_llm: AsyncMock):
    """Summary parsing should still work in pull mode."""
    mock_llm.side_effect = [
        make_mock_response(content="<summary>A summary.</summary>\nA detailed output."),
        make_mock_response(content="B output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="pull")
    result = await swarm.run("Task")

    assert result.history[0].output == "A detailed output."
    assert result.history[0].summary == "A summary."


async def test_pull_mode_parallel_step(mock_llm: AsyncMock):
    """Pull mode works with parallel steps."""
    mock_llm.side_effect = [
        make_mock_response(content="A output."),
        make_mock_response(content="B output."),
        make_mock_response(content="C output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    swarm = Swarm(flow=chain(a, parallel(b, c)), context_mode="pull")
    result = await swarm.run("Task")

    assert result.context["a"] == "A output."
    assert result.context["b"] == "B output."
    assert result.context["c"] == "C output."

    # B and C should have context tools
    for call in mock_llm.call_args_list[1:]:
        tools = call.kwargs.get("tools", [])
        tool_names = {t["function"]["name"] for t in tools}
        assert "get_context" in tool_names


async def test_pull_mode_default(mock_llm: AsyncMock):
    """Default context_mode should be pull."""
    mock_llm.side_effect = [
        make_mock_response(content="A output."),
        make_mock_response(content="B output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b)
    await swarm.run("Task")

    # Should behave as pull mode: second agent gets context tools
    b_call = mock_llm.call_args_list[1]
    tools = b_call.kwargs.get("tools", [])
    tool_names = {t["function"]["name"] for t in tools}
    assert "get_context" in tool_names
    assert "expand_context" not in tool_names
