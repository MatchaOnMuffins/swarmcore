import pytest

from swarmcore.exceptions import SwarmError
from swarmcore.swarm import parse_flow


def test_single_agent():
    assert parse_flow("researcher") == ["researcher"]


def test_sequential_chain():
    assert parse_flow("a >> b >> c") == ["a", "b", "c"]


def test_parallel_group():
    assert parse_flow("[a, b]") == [["a", "b"]]


def test_mixed_flow():
    assert parse_flow("a >> [b, c] >> d") == ["a", ["b", "c"], "d"]


def test_whitespace_handling():
    assert parse_flow("  a  >>  [ b , c ]  >>  d  ") == ["a", ["b", "c"], "d"]


def test_multiple_parallel_groups():
    result = parse_flow("[a, b] >> [c, d]")
    assert result == [["a", "b"], ["c", "d"]]


def test_empty_flow_raises():
    with pytest.raises(SwarmError, match="empty"):
        parse_flow("")


def test_whitespace_only_flow_raises():
    with pytest.raises(SwarmError, match="empty"):
        parse_flow("   ")


def test_empty_step_raises():
    with pytest.raises(SwarmError, match="empty step"):
        parse_flow("a >> >> b")


def test_empty_parallel_group_raises():
    with pytest.raises(SwarmError, match="empty parallel group"):
        parse_flow("a >> [] >> b")


def test_malformed_brackets_raises():
    with pytest.raises(SwarmError, match="Malformed"):
        parse_flow("a >> [b, c >> d")
