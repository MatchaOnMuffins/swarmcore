from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


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


async def test_pull_mode_prev_step_pushed_no_tools(mock_llm: AsyncMock):
    """In pull mode A >> B, B gets A's full output pushed — no pull tools needed."""
    mock_llm.side_effect = [
        make_mock_response(content="A output."),
        make_mock_response(content="B output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="pull")
    await swarm.run("Task")

    # B's call should NOT have any context tools (A is immediately preceding)
    b_call = mock_llm.call_args_list[1]
    tools = b_call.kwargs.get("tools", [])
    tool_names = {t["function"]["name"] for t in tools}
    assert "list_context" not in tool_names
    assert "get_context" not in tool_names
    assert "search_context" not in tool_names
    assert "expand_context" not in tool_names


async def test_pull_mode_pushes_prev_step_full_output(mock_llm: AsyncMock):
    """Pull mode pushes immediately preceding agent's full output into prompt."""
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
    # Should contain A's full detail (pushed, not pulled)
    assert "Detailed research output with many words." in b_system
    # Should contain the agent name
    assert "a" in b_system


async def test_pull_mode_three_step_hybrid(mock_llm: AsyncMock):
    """A >> B >> C: C gets B's full output pushed, A available via pull tools."""
    mock_llm.side_effect = [
        make_mock_response(
            content="<summary>A summary.</summary>\nA detailed output."
        ),
        make_mock_response(
            content="<summary>B summary.</summary>\nB detailed output."
        ),
        make_mock_response(content="C output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    swarm = Swarm(flow=a >> b >> c, context_mode="pull")
    await swarm.run("Task")

    c_call = mock_llm.call_args_list[2]
    c_system = c_call.kwargs["messages"][0]["content"]
    # C should have B's full output pushed
    assert "B detailed output." in c_system
    # C should NOT have A's full output (only summary via pull tools)
    assert "A detailed output." not in c_system
    assert "A summary." in c_system
    # C should have pull tools for earlier agent A
    tools = c_call.kwargs.get("tools", [])
    tool_names = {t["function"]["name"] for t in tools}
    assert "get_context" in tool_names


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


async def test_pull_mode_agent_calls_get_context_for_earlier(mock_llm: AsyncMock):
    """A >> B >> C: C can call get_context to retrieve earlier agent A's full output."""
    tool_call = MagicMock()
    tool_call.id = "call_get"
    tool_call.function.name = "get_context"
    tool_call.function.arguments = '{"agent_name": "a"}'

    mock_llm.side_effect = [
        make_mock_response(content="A detailed output."),
        make_mock_response(content="B output."),
        # C first calls get_context for earlier agent A, then produces final output
        make_mock_response(content=None, tool_calls=[tool_call]),
        make_mock_response(content="C output using A's data."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    swarm = Swarm(flow=a >> b >> c, context_mode="pull")
    result = await swarm.run("Task")

    assert result.output == "C output using A's data."

    # C should have recorded a tool call for get_context
    c_result = result.history[2]
    assert c_result.tool_call_count == 1
    assert c_result.tool_calls[0].tool_name == "get_context"
    assert "A detailed output." in c_result.tool_calls[0].result


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
    """Pull mode works with parallel steps — prev step output is pushed."""
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

    # B and C both get A's full output pushed (A is the prev step)
    for call in mock_llm.call_args_list[1:]:
        system = call.kwargs["messages"][0]["content"]
        assert "A output." in system


# --- Nested sub-flow execution tests ---


async def test_nested_subchains_execution_order(mock_llm: AsyncMock):
    """(a >> b) | (c >> d): a runs before b, c before d, branches concurrent."""
    call_order: list[str] = []

    async def track_calls(**kwargs):
        messages = kwargs["messages"]
        system = messages[0]["content"]
        # Identify which agent this is by instructions
        if "Do A." in system:
            call_order.append("a")
            return make_mock_response(content="A output")
        elif "Do B." in system:
            call_order.append("b")
            return make_mock_response(content="B output")
        elif "Do C." in system:
            call_order.append("c")
            return make_mock_response(content="C output")
        elif "Do D." in system:
            call_order.append("d")
            return make_mock_response(content="D output")
        return make_mock_response(content="Unknown")

    mock_llm.side_effect = track_calls

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")
    d = Agent(name="d", instructions="Do D.")

    swarm = Swarm(flow=(a >> b) | (c >> d), context_mode="pull")
    result = await swarm.run("Task")

    assert len(result.history) == 4
    assert result.context["a"] == "A output"
    assert result.context["b"] == "B output"
    assert result.context["c"] == "C output"
    assert result.context["d"] == "D output"

    # a must come before b, c must come before d
    assert call_order.index("a") < call_order.index("b")
    assert call_order.index("c") < call_order.index("d")


async def test_nested_subchain_context_sharing(mock_llm: AsyncMock):
    """Within sub-chain a >> b, b sees a's output in context."""

    async def route(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "Do A." in system:
            return make_mock_response(content="A output")
        if "Do B." in system:
            return make_mock_response(content="B output")
        if "Do C." in system:
            return make_mock_response(content="C output")
        return make_mock_response(content="Unknown")

    mock_llm.side_effect = route

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    swarm = Swarm(flow=(a >> b) | c, context_mode="pull")
    result = await swarm.run("Task")

    # b should have context tools available (since a ran before it)
    assert result.context["a"] == "A output"
    assert result.context["b"] == "B output"
    assert result.context["c"] == "C output"


async def test_nested_prev_step_names_after_parallel(mock_llm: AsyncMock):
    """After (a>>b) | (c>>d), next agent's prev_step_names = {"b", "d"}."""

    async def route(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "Do A." in system:
            return make_mock_response(content="A output")
        if "Do B." in system:
            return make_mock_response(content="B output")
        if "Do C." in system:
            return make_mock_response(content="C output")
        if "Do D." in system:
            return make_mock_response(content="D output")
        if "Write." in system:
            return make_mock_response(content="Writer output")
        return make_mock_response(content="Unknown")

    mock_llm.side_effect = route

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")
    d = Agent(name="d", instructions="Do D.")
    writer = Agent(name="writer", instructions="Write.")

    swarm = Swarm(flow=((a >> b) | (c >> d)) >> writer, context_mode="push")
    result = await swarm.run("Task")

    assert result.output == "Writer output"
    assert len(result.history) == 5

    # Writer's system prompt should have b and d's full output (as prev step)
    writer_call = mock_llm.call_args_list[-1]
    writer_system = writer_call.kwargs["messages"][0]["content"]
    assert "B output" in writer_system
    assert "D output" in writer_system


async def test_nested_mixed_parallel_group(mock_llm: AsyncMock):
    """Mixed parallel group: bare Agent + sub-Flow in same parallel step."""

    async def route(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "Do A." in system:
            return make_mock_response(content="A output")
        if "Do B." in system:
            return make_mock_response(content="B output")
        if "Do C." in system:
            return make_mock_response(content="C output")
        return make_mock_response(content="Unknown")

    mock_llm.side_effect = route

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    # c is bare agent, a >> b is a sub-flow
    swarm = Swarm(flow=(a >> b) | c, context_mode="pull")
    result = await swarm.run("Task")

    assert result.context["a"] == "A output"
    assert result.context["b"] == "B output"
    assert result.context["c"] == "C output"


async def test_nested_token_accumulation(mock_llm: AsyncMock):
    """Token accumulation includes all sub-flow agents."""

    async def route(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "Do A." in system:
            return make_mock_response(
                content="A output", prompt_tokens=5, completion_tokens=10
            )
        if "Do B." in system:
            return make_mock_response(
                content="B output", prompt_tokens=8, completion_tokens=12
            )
        if "Do C." in system:
            return make_mock_response(
                content="C output", prompt_tokens=3, completion_tokens=7
            )
        return make_mock_response(content="Unknown")

    mock_llm.side_effect = route

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    swarm = Swarm(flow=(a >> b) | c, context_mode="pull")
    result = await swarm.run("Task")

    assert result.total_token_usage.prompt_tokens == 16  # 5 + 3 + 8
    assert result.total_token_usage.completion_tokens == 29  # 10 + 7 + 12
    assert result.total_token_usage.total_tokens == 45


async def test_parallel_final_step_combines_outputs(mock_llm: AsyncMock):
    """When a parallel group is the final step, result.output includes all agents."""
    mock_llm.side_effect = [
        make_mock_response(content="Agent A output"),
        make_mock_response(content="Agent B output"),
        make_mock_response(content="Agent C output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")
    c = Agent(name="c", instructions="Do C.")

    swarm = Swarm(flow=chain(parallel(a, b, c)), context_mode="push")
    result = await swarm.run("Test task")

    # All three agents' outputs should appear in result.output
    assert "Agent A output" in result.output
    assert "Agent B output" in result.output
    assert "Agent C output" in result.output
    # Each section should be headed with the agent name
    assert "## a" in result.output
    assert "## b" in result.output
    assert "## c" in result.output


async def test_parallel_final_step_pull_mode(mock_llm: AsyncMock):
    """Parallel final step combines outputs in pull mode too."""
    mock_llm.side_effect = [
        make_mock_response(content="Alpha output"),
        make_mock_response(content="Beta output"),
    ]

    alpha = Agent(name="alpha", instructions="Do Alpha.")
    beta = Agent(name="beta", instructions="Do Beta.")

    swarm = Swarm(flow=chain(parallel(alpha, beta)), context_mode="pull")
    result = await swarm.run("Test task")

    assert "Alpha output" in result.output
    assert "Beta output" in result.output
    assert "## alpha" in result.output
    assert "## beta" in result.output


async def test_parallel_final_step_no_duplicate_from_earlier_step(mock_llm: AsyncMock):
    """Agent reused across steps should only appear once in final merged output.

    Regression: chain(a, parallel(a, b)) should NOT include the first step's
    output for 'a' in the merged output — only the parallel step's result.
    """
    mock_llm.side_effect = [
        make_mock_response(content="A step-1 output"),
        make_mock_response(content="A step-2 output"),
        make_mock_response(content="B output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=chain(a, parallel(a, b)), context_mode="pull")
    result = await swarm.run("Test task")

    assert "A step-2 output" in result.output
    assert "B output" in result.output
    assert "A step-1 output" not in result.output


async def test_parallel_final_step_no_duplicate_push_mode(mock_llm: AsyncMock):
    """Same duplicate-prevention check in push mode."""
    mock_llm.side_effect = [
        make_mock_response(content="A step-1 output"),
        make_mock_response(content="A step-2 output"),
        make_mock_response(content="B output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=chain(a, parallel(a, b)), context_mode="push")
    result = await swarm.run("Test task")

    assert "A step-2 output" in result.output
    assert "B output" in result.output
    assert "A step-1 output" not in result.output


async def test_sequential_final_step_unchanged(mock_llm: AsyncMock):
    """Sequential final step still returns only the last agent's output."""
    mock_llm.side_effect = [
        make_mock_response(content="First output"),
        make_mock_response(content="Final output"),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b, context_mode="push")
    result = await swarm.run("Test task")

    assert result.output == "Final output"


async def test_pull_mode_default(mock_llm: AsyncMock):
    """Default context_mode should be pull with prev-step pushing."""
    mock_llm.side_effect = [
        make_mock_response(content="A output."),
        make_mock_response(content="B output."),
    ]

    a = Agent(name="a", instructions="Do A.")
    b = Agent(name="b", instructions="Do B.")

    swarm = Swarm(flow=a >> b)
    await swarm.run("Task")

    # Should behave as pull mode: A's output pushed to B (no tools needed)
    b_call = mock_llm.call_args_list[1]
    tools = b_call.kwargs.get("tools", [])
    tool_names = {t["function"]["name"] for t in tools}
    assert "get_context" not in tool_names
    assert "expand_context" not in tool_names
    # A's full output should be in B's system prompt
    b_system = b_call.kwargs["messages"][0]["content"]
    assert "A output." in b_system
