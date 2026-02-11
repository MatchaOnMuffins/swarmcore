"""Tests for pre-built agent factories."""

from __future__ import annotations

from swarmcore.agent import Agent
from swarmcore.agents import analyst, editor, researcher, summarizer, writer
from swarmcore.flow import Flow
from swarmcore.tools import search_web


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------


class TestDefaultConstruction:
    def test_researcher_defaults(self) -> None:
        agent = researcher()
        assert agent.name == "researcher"
        assert "research" in agent.instructions.lower()
        assert agent.model == "anthropic/claude-opus-4-6"

    def test_analyst_defaults(self) -> None:
        agent = analyst()
        assert agent.name == "analyst"
        assert "analy" in agent.instructions.lower()
        assert agent.model == "anthropic/claude-opus-4-6"

    def test_writer_defaults(self) -> None:
        agent = writer()
        assert agent.name == "writer"
        assert "writ" in agent.instructions.lower()
        assert agent.model == "anthropic/claude-opus-4-6"

    def test_editor_defaults(self) -> None:
        agent = editor()
        assert agent.name == "editor"
        assert "editor" in agent.instructions.lower()
        assert agent.model == "anthropic/claude-opus-4-6"

    def test_summarizer_defaults(self) -> None:
        agent = summarizer()
        assert agent.name == "summarizer"
        assert "summar" in agent.instructions.lower()
        assert agent.model == "anthropic/claude-opus-4-6"


# ---------------------------------------------------------------------------
# Default tools
# ---------------------------------------------------------------------------


class TestDefaultTools:
    def test_researcher_has_search_web(self) -> None:
        agent = researcher()
        assert "search_web" in agent._tools
        assert agent._tools["search_web"] is search_web
        assert len(agent._tool_schemas) == 1

    def test_analyst_has_no_tools(self) -> None:
        agent = analyst()
        assert agent._tools == {}
        assert agent._tool_schemas == []

    def test_writer_has_no_tools(self) -> None:
        agent = writer()
        assert agent._tools == {}
        assert agent._tool_schemas == []

    def test_editor_has_no_tools(self) -> None:
        agent = editor()
        assert agent._tools == {}
        assert agent._tool_schemas == []

    def test_summarizer_has_no_tools(self) -> None:
        agent = summarizer()
        assert agent._tools == {}
        assert agent._tool_schemas == []


# ---------------------------------------------------------------------------
# Parameter overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    def test_custom_name(self) -> None:
        agent = researcher(name="market_researcher")
        assert agent.name == "market_researcher"

    def test_custom_instructions(self) -> None:
        agent = writer(instructions="Write haiku only.")
        assert agent.instructions == "Write haiku only."

    def test_custom_model(self) -> None:
        agent = analyst(model="openai/gpt-4o")
        assert agent.model == "openai/gpt-4o"

    def test_custom_tools(self) -> None:
        def my_tool(x: str) -> str:
            """A custom tool."""
            return x

        agent = researcher(tools=[my_tool])
        assert "my_tool" in agent._tools
        assert "search_web" not in agent._tools

    def test_tools_empty_list_removes_defaults(self) -> None:
        agent = researcher(tools=[])
        assert agent._tools == {}
        assert agent._tool_schemas == []

    def test_tools_none_removes_defaults(self) -> None:
        agent = researcher(tools=None)
        assert agent._tools == {}
        assert agent._tool_schemas == []

    def test_timeout_forwarded(self) -> None:
        agent = researcher(timeout=30.0)
        assert agent.timeout == 30.0

    def test_max_retries_forwarded(self) -> None:
        agent = analyst(max_retries=5)
        assert agent.max_retries == 5

    def test_max_turns_forwarded(self) -> None:
        agent = writer(max_turns=3)
        assert agent.max_turns == 3


# ---------------------------------------------------------------------------
# Flow composition
# ---------------------------------------------------------------------------


class TestFlowComposition:
    def test_sequential_flow(self) -> None:
        flow = researcher() >> analyst() >> editor()
        assert isinstance(flow, Flow)

    def test_parallel_flow(self) -> None:
        flow = analyst() | writer()
        assert isinstance(flow, Flow)

    def test_mixed_flow(self) -> None:
        flow = researcher() >> (analyst() | writer()) >> editor()
        assert isinstance(flow, Flow)


# ---------------------------------------------------------------------------
# Fresh instances
# ---------------------------------------------------------------------------


class TestFreshInstances:
    def test_each_call_returns_new_instance(self) -> None:
        a = researcher()
        b = researcher()
        assert a is not b

    def test_all_factories_return_agent(self) -> None:
        for factory in [researcher, analyst, writer, editor, summarizer]:
            agent = factory()
            assert isinstance(agent, Agent)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_importable_from_swarmcore(self) -> None:
        import swarmcore

        assert hasattr(swarmcore, "researcher")
        assert hasattr(swarmcore, "analyst")
        assert hasattr(swarmcore, "writer")
        assert hasattr(swarmcore, "editor")
        assert hasattr(swarmcore, "summarizer")

    def test_in_all(self) -> None:
        import swarmcore

        for name in ["researcher", "analyst", "writer", "editor", "summarizer"]:
            assert name in swarmcore.__all__
