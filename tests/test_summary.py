from __future__ import annotations

import io

import pytest

from swarmcore.models import AgentResult, SwarmResult, TokenUsage, ToolCallRecord


def _agent(
    name: str,
    *,
    tokens: int = 100,
    calls: int = 1,
    tools: int = 0,
    duration: float = 1.0,
    tool_calls: list[ToolCallRecord] | None = None,
    cost: float = 0.0,
) -> AgentResult:
    return AgentResult(
        agent_name=name,
        input_task="test task",
        output=f"output from {name}",
        model="test/model",
        duration_seconds=duration,
        token_usage=TokenUsage(
            prompt_tokens=tokens // 2,
            completion_tokens=tokens - tokens // 2,
            total_tokens=tokens,
        ),
        llm_call_count=calls,
        tool_call_count=tools,
        tool_calls=tool_calls or [],
        cost=cost,
    )


def _swarm_result(agents: list[AgentResult], *, duration: float = 10.0) -> SwarmResult:
    total = TokenUsage()
    total_cost = 0.0
    for a in agents:
        total.prompt_tokens += a.token_usage.prompt_tokens
        total.completion_tokens += a.token_usage.completion_tokens
        total.total_tokens += a.token_usage.total_tokens
        total_cost += a.cost

    return SwarmResult(
        output=agents[-1].output if agents else "",
        context={a.agent_name: a.output for a in agents},
        history=agents,
        duration_seconds=duration,
        total_token_usage=total,
        total_cost=total_cost,
    )


# -- token_usage_table ------------------------------------------------


def test_token_usage_table_basic():
    result = _swarm_result(
        [
            _agent("researcher", tokens=1000, calls=2, tools=1, duration=5.0),
            _agent("writer", tokens=2000, calls=1, tools=0, duration=3.0),
        ],
        duration=8.0,
    )

    table = result.token_usage_table()
    assert "researcher" in table
    assert "writer" in table
    assert "TOTAL" in table
    assert "1,000" in table
    assert "2,000" in table
    assert "3,000" in table


def test_token_usage_table_column_alignment():
    """Agent names of different lengths should be aligned."""
    result = _swarm_result(
        [
            _agent("a", tokens=100),
            _agent("very_long_agent_name", tokens=200),
        ]
    )

    table = result.token_usage_table()
    lines = table.strip().split("\n")
    # All data lines should have the same indentation pattern
    # (header, rule, agent1, agent2, rule, total = 6 lines)
    assert len(lines) == 6


def test_token_usage_table_empty_history():
    result = _swarm_result([])
    assert result.token_usage_table() == ""


def test_token_usage_table_with_cost():
    """Agents with non-zero cost show the cost column."""
    result = _swarm_result(
        [
            _agent(
                "fusion_physicist",
                tokens=4727,
                calls=2,
                tools=1,
                duration=12.1,
                cost=0.0892,
            ),
            _agent(
                "grid_engineer",
                tokens=3891,
                calls=1,
                tools=0,
                duration=6.8,
                cost=0.0734,
            ),
        ],
        duration=43.9,
    )

    table = result.token_usage_table()
    assert "Cost" in table
    assert "$0.0892" in table
    assert "$0.0734" in table
    # Total cost should be present
    total_cost_str = f"${result.total_cost:.4f}"
    assert total_cost_str in table


def test_token_usage_table_no_cost():
    """Agents with zero cost (default) don't show cost column."""
    result = _swarm_result(
        [
            _agent("researcher", tokens=1000, calls=2, tools=1, duration=5.0),
            _agent("writer", tokens=2000, calls=1, tools=0, duration=3.0),
        ],
        duration=8.0,
    )

    table = result.token_usage_table()
    assert "Cost" not in table
    assert "$" not in table


def test_total_cost_field():
    """Verify SwarmResult.total_cost works."""
    result = _swarm_result(
        [
            _agent("a", cost=0.05),
            _agent("b", cost=0.10),
            _agent("c", cost=0.15),
        ]
    )
    assert result.total_cost == pytest.approx(0.30)


def test_cost_field_on_agent_result():
    """Verify AgentResult.cost field works."""
    agent = _agent("test_agent", cost=0.1234)
    assert agent.cost == 0.1234

    # Default cost is 0.0
    agent_default = _agent("default_agent")
    assert agent_default.cost == 0.0


def test_cost_column_formatting():
    """Verify dollar sign, decimal places in cost column."""
    result = _swarm_result(
        [
            _agent("agent_a", tokens=500, cost=0.0001),
            _agent("agent_b", tokens=1500, cost=1.2345),
        ],
        duration=5.0,
    )

    table = result.token_usage_table()
    # Cost should have dollar sign and 4 decimal places
    assert "$0.0001" in table
    assert "$1.2345" in table
    # Total cost
    assert "$1.2346" in table
    # Header should contain "Cost"
    header_line = table.split("\n")[0]
    assert "Cost" in header_line


# -- context_pull_report -----------------------------------------------


def _get_context_call(agent_name: str) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name="get_context",
        arguments={"agent_name": agent_name},
        result="...",
        duration_seconds=0.01,
    )


def test_context_pull_report_selective():
    """Agent that pulls 2 out of 3 prior agents is marked SELECTIVE."""
    result = _swarm_result(
        [
            _agent("a"),
            _agent("b"),
            _agent("c"),
            _agent(
                "synthesizer",
                tool_calls=[_get_context_call("a"), _get_context_call("b")],
                tools=2,
            ),
        ]
    )

    report = result.context_pull_report()
    assert "SELECTIVE" in report
    assert "2/3" in report
    assert "skipped: c" in report


def test_context_pull_report_pulled_all():
    """Agent that pulls all prior agents is marked PULLED ALL."""
    result = _swarm_result(
        [
            _agent("a"),
            _agent("b"),
            _agent(
                "synthesizer",
                tool_calls=[_get_context_call("a"), _get_context_call("b")],
                tools=2,
            ),
        ]
    )

    report = result.context_pull_report()
    assert "PULLED ALL" in report
    assert "2/2" in report


def test_context_pull_report_no_pulls():
    """No agents used get_context — empty report."""
    result = _swarm_result([_agent("a"), _agent("b")])
    assert result.context_pull_report() == ""


def test_context_pull_report_first_agent_skipped():
    """First-tier agents that have no prior agents produce no report line."""
    result = _swarm_result(
        [
            _agent("a", tool_calls=[_get_context_call("nonexistent")]),
            _agent("b"),
        ]
    )

    report = result.context_pull_report()
    # "a" is index 0, so it has no prior agents — should be skipped
    assert "a" not in report or report == ""


def test_context_pull_report_multiple_agents():
    """Multiple agents pulling context each get their own line."""
    result = _swarm_result(
        [
            _agent("a"),
            _agent("b"),
            _agent("c"),
            _agent(
                "d",
                tool_calls=[_get_context_call("a")],
                tools=1,
            ),
            _agent(
                "e",
                tool_calls=[_get_context_call("a"), _get_context_call("d")],
                tools=2,
            ),
        ]
    )

    report = result.context_pull_report()
    lines = [line for line in report.strip().split("\n") if line.strip()]
    assert len(lines) == 2


# -- summary -----------------------------------------------------------


def test_summary_includes_both_sections():
    result = _swarm_result(
        [
            _agent("a"),
            _agent("b", tool_calls=[_get_context_call("a")], tools=1),
        ]
    )

    text = result.summary()
    assert "Token Usage" in text
    assert "Context Pull Analysis" in text


def test_summary_no_pulls_omits_context_section():
    result = _swarm_result([_agent("a"), _agent("b")])

    text = result.summary()
    assert "Token Usage" in text
    assert "Context Pull Analysis" not in text


def test_summary_empty_history():
    result = _swarm_result([])
    assert result.summary() == ""


# -- print_summary -----------------------------------------------------


def test_print_summary_writes_to_file():
    result = _swarm_result(
        [_agent("a", tokens=500), _agent("b", tokens=300)],
        duration=5.0,
    )

    buf = io.StringIO()
    result.print_summary(file=buf)

    output = buf.getvalue()
    assert "Token Usage" in output
    assert "TOTAL" in output


def test_print_summary_empty_history_writes_nothing():
    result = _swarm_result([])

    buf = io.StringIO()
    result.print_summary(file=buf)

    assert buf.getvalue() == ""
