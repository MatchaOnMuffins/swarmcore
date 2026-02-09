from swarmcore.context import SharedContext
from swarmcore.context_tools import (
    make_context_tools,
    make_get_context_tool,
    make_list_context_tool,
    make_search_context_tool,
)


# --- list_context ---


def test_list_context_empty():
    ctx = SharedContext()
    tool = make_list_context_tool(ctx)
    assert tool() == "No agent outputs available yet."


def test_list_context_populated():
    ctx = SharedContext()
    ctx.set("researcher", "Full research output here.", summary="Research summary.")
    ctx.set("critic", "Full critique.", summary="Critique summary.")
    result = make_list_context_tool(ctx)()
    assert "researcher" in result
    assert "Research summary." in result
    assert "critic" in result
    assert "Critique summary." in result
    assert "chars" in result


# --- get_context ---


def test_get_context_existing():
    ctx = SharedContext()
    ctx.set("researcher", "Detailed research output.")
    tool = make_get_context_tool(ctx)
    assert tool("researcher") == "Detailed research output."


def test_get_context_missing():
    ctx = SharedContext()
    ctx.set("researcher", "Some output")
    tool = make_get_context_tool(ctx)
    result = tool("nonexistent")
    assert "No context found" in result
    assert "researcher" in result


def test_get_context_no_agents():
    ctx = SharedContext()
    tool = make_get_context_tool(ctx)
    result = tool("anyone")
    assert result == "No agent outputs available yet."


# --- search_context ---


def test_search_context_matches():
    ctx = SharedContext()
    ctx.set("researcher", "AI trends are growing.\nMarket is expanding.")
    ctx.set("critic", "AI trends are overhyped.")
    tool = make_search_context_tool(ctx)
    result = tool("AI trends")
    assert "researcher" in result
    assert "critic" in result
    assert "AI trends are growing." in result
    assert "AI trends are overhyped." in result


def test_search_context_no_matches():
    ctx = SharedContext()
    ctx.set("agent", "Some output")
    tool = make_search_context_tool(ctx)
    result = tool("nonexistent")
    assert "No matches found" in result


def test_search_context_regex():
    ctx = SharedContext()
    ctx.set("agent", "foo123bar\nother line")
    tool = make_search_context_tool(ctx)
    result = tool(r"foo\d+bar")
    assert "foo123bar" in result


def test_search_context_invalid_regex_fallback():
    ctx = SharedContext()
    ctx.set("agent", "value is [bracket]")
    tool = make_search_context_tool(ctx)
    result = tool("[bracket]")
    assert "[bracket]" in result


# --- make_context_tools ---


def test_make_context_tools_returns_three():
    ctx = SharedContext()
    tools = make_context_tools(ctx)
    assert len(tools) == 3
    names = {t.__name__ for t in tools}
    assert names == {"list_context", "get_context", "search_context"}
