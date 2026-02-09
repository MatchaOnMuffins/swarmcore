from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from swarmcore import Agent, Swarm, SwarmResult
from swarmcore.exceptions import SwarmError
from swarmcore.models import AgentResult
from tests.conftest import make_mock_response


async def test_sequential_flow(mock_llm: AsyncMock):
    mock_llm.side_effect = [
        make_mock_response(content="Research output"),
        make_mock_response(content="Writer output"),
    ]

    swarm = Swarm(
        agents=[
            Agent(name="researcher", instructions="Research."),
            Agent(name="writer", instructions="Write."),
        ],
        flow="researcher >> writer",
    )

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

    swarm = Swarm(
        agents=[
            Agent(name="a", instructions="Do A."),
            Agent(name="b", instructions="Do B."),
        ],
        flow="[a, b]",
    )

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

    swarm = Swarm(
        agents=[
            Agent(name="planner", instructions="Plan."),
            Agent(name="researcher", instructions="Research."),
            Agent(name="critic", instructions="Critique."),
            Agent(name="writer", instructions="Write."),
        ],
        flow="planner >> [researcher, critic] >> writer",
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

    swarm = Swarm(
        agents=[
            Agent(name="researcher", instructions="Research."),
            Agent(name="writer", instructions="Write."),
        ],
        flow="researcher >> writer",
    )

    await swarm.run("Test task")

    # The second call (writer) should have context from the first (researcher)
    second_call = mock_llm.call_args_list[1]
    system_msg = second_call.kwargs["messages"][0]["content"]
    assert "researcher" in system_msg
    assert "Research notes here" in system_msg


def test_invalid_agent_name_raises():
    with pytest.raises(SwarmError, match="nonexistent"):
        Swarm(
            agents=[Agent(name="a", instructions="Do A.")],
            flow="a >> nonexistent",
        )


async def test_result_structure(mock_llm: AsyncMock):
    swarm = Swarm(
        agents=[Agent(name="solo", instructions="Work alone.")],
        flow="solo",
    )

    result = await swarm.run("Do it")

    assert isinstance(result, SwarmResult)
    assert isinstance(result.output, str)
    assert isinstance(result.context, dict)
    assert isinstance(result.history, list)
    assert all(isinstance(r, AgentResult) for r in result.history)
