from __future__ import annotations

import pytest

from swarmcore import Agent, Flow, chain, parallel
from swarmcore.exceptions import SwarmError


@pytest.fixture
def a() -> Agent:
    return Agent(name="a", instructions="Do A.")


@pytest.fixture
def b() -> Agent:
    return Agent(name="b", instructions="Do B.")


@pytest.fixture
def c() -> Agent:
    return Agent(name="c", instructions="Do C.")


@pytest.fixture
def d() -> Agent:
    return Agent(name="d", instructions="Do D.")


# --- chain() ---


def test_chain_single_agent(a: Agent):
    flow = chain(a)
    assert len(flow.steps) == 1
    assert flow.steps[0] is a


def test_chain_multiple_agents(a: Agent, b: Agent, c: Agent):
    flow = chain(a, b, c)
    assert len(flow.steps) == 3
    assert flow.steps[0] is a
    assert flow.steps[1] is b
    assert flow.steps[2] is c


def test_chain_with_parallel(a: Agent, b: Agent, c: Agent, d: Agent):
    flow = chain(a, parallel(b, c), d)
    assert len(flow.steps) == 3
    assert flow.steps[0] is a
    assert isinstance(flow.steps[1], list)
    assert flow.steps[1][0] is b
    assert flow.steps[1][1] is c
    assert flow.steps[2] is d


def test_chain_empty_raises():
    with pytest.raises(SwarmError, match="at least one"):
        chain()


# --- parallel() ---


def test_parallel_two_agents(a: Agent, b: Agent):
    group = parallel(a, b)
    assert group.agents == [a, b]


def test_parallel_fewer_than_two_raises(a: Agent):
    with pytest.raises(SwarmError, match="at least 2"):
        parallel(a)


# --- >> operator ---


def test_rshift_agent_agent(a: Agent, b: Agent):
    flow = a >> b
    assert len(flow.steps) == 2
    assert flow.steps[0] is a
    assert flow.steps[1] is b


def test_rshift_agent_flow(a: Agent, b: Agent, c: Agent):
    flow = a >> (b >> c)
    assert len(flow.steps) == 3
    assert flow.steps[0] is a
    assert flow.steps[1] is b
    assert flow.steps[2] is c


def test_rshift_flow_agent(a: Agent, b: Agent, c: Agent):
    flow = (a >> b) >> c
    assert len(flow.steps) == 3


def test_rshift_flow_flow(a: Agent, b: Agent, c: Agent, d: Agent):
    flow = (a >> b) >> (c >> d)
    assert len(flow.steps) == 4


# --- | operator ---


def test_or_agent_agent(a: Agent, b: Agent):
    flow = a | b
    assert len(flow.steps) == 1
    assert isinstance(flow.steps[0], list)
    assert len(flow.steps[0]) == 2


def test_or_agent_flow(a: Agent, b: Agent, c: Agent):
    # a | (b | c) should produce a single parallel group with all three
    flow = a | (b | c)
    assert len(flow.steps) == 1
    assert isinstance(flow.steps[0], list)
    assert len(flow.steps[0]) == 3


def test_or_flow_agent(a: Agent, b: Agent, c: Agent):
    # (a | b) | c should extend the parallel group
    flow = (a | b) | c
    assert len(flow.steps) == 1
    assert isinstance(flow.steps[0], list)
    assert len(flow.steps[0]) == 3


# --- Mixed operators ---


def test_mixed_rshift_and_or(a: Agent, b: Agent, c: Agent, d: Agent):
    flow = a >> (b | c) >> d
    assert len(flow.steps) == 3
    assert flow.steps[0] is a
    assert isinstance(flow.steps[1], list)
    assert len(flow.steps[1]) == 2
    assert flow.steps[2] is d


# --- Properties ---


def test_agents_property(a: Agent, b: Agent, c: Agent, d: Agent):
    flow = chain(a, parallel(b, c), d)
    agents = flow.agents
    assert [ag.name for ag in agents] == ["a", "b", "c", "d"]


def test_agents_deduplicates(a: Agent, b: Agent):
    # Same agent in two steps should appear once
    flow = Flow([a, b, a])
    agents = flow.agents
    assert [ag.name for ag in agents] == ["a", "b"]


# --- repr ---


def test_repr_sequential(a: Agent, b: Agent, c: Agent):
    flow = chain(a, b, c)
    assert repr(flow) == "Flow(a >> b >> c)"


def test_repr_parallel(a: Agent, b: Agent):
    flow = chain(parallel(a, b))
    assert repr(flow) == "Flow([a, b])"


def test_repr_mixed(a: Agent, b: Agent, c: Agent, d: Agent):
    flow = chain(a, parallel(b, c), d)
    assert repr(flow) == "Flow(a >> [b, c] >> d)"
