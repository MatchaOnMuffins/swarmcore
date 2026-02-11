from __future__ import annotations

from unittest.mock import MagicMock, patch

from swarmcore.agent import _function_to_tool_schema
from swarmcore.tools import search_web


# --- search_web returns formatted results ---


def test_search_web_returns_formatted_results():
    fake_results = [
        {
            "title": "First Result",
            "body": "Description of first result.",
            "href": "https://example.com/1",
        },
        {
            "title": "Second Result",
            "body": "Description of second result.",
            "href": "https://example.com/2",
        },
    ]
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = fake_results

    mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

    with patch.dict(
        "sys.modules", {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}
    ):
        # Re-import to pick up the patched module
        from importlib import reload

        import swarmcore.tools

        reload(swarmcore.tools)
        result = swarmcore.tools.search_web("test query")

    assert "**First Result**" in result
    assert "Description of first result." in result
    assert "https://example.com/1" in result
    assert "**Second Result**" in result
    assert "Description of second result." in result
    assert "https://example.com/2" in result
    # Two results separated by double newline
    assert result.count("\n\n") == 1


# --- max_results is forwarded ---


def test_search_web_forwards_max_results():
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = []

    mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

    with patch.dict(
        "sys.modules", {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}
    ):
        from importlib import reload

        import swarmcore.tools

        reload(swarmcore.tools)
        swarmcore.tools.search_web("test query", max_results=10)

    mock_ddgs_instance.text.assert_called_once_with("test query", max_results=10)


# --- graceful degradation when ddgs is not installed ---


def test_search_web_import_error():
    original_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def mock_import(name, *args, **kwargs):
        if name == "duckduckgo_search":
            raise ImportError("No module named 'duckduckgo_search'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        # Need to reload to trigger the lazy import path fresh
        from importlib import reload

        import swarmcore.tools

        reload(swarmcore.tools)
        result = swarmcore.tools.search_web("test query")

    assert result == (
        "Error: duckduckgo-search is not installed. "
        "Install it with: pip install duckduckgo-search"
    )


# --- empty results ---


def test_search_web_empty_results():
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = []

    mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

    with patch.dict(
        "sys.modules", {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}
    ):
        from importlib import reload

        import swarmcore.tools

        reload(swarmcore.tools)
        result = swarmcore.tools.search_web("nonexistent query")

    assert result == "No results found."


# --- tool schema compatibility ---


def test_search_web_tool_schema():
    schema = _function_to_tool_schema(search_web)

    assert schema["type"] == "function"
    func = schema["function"]
    assert func["name"] == "search_web"
    assert "Search the web" in func["description"]

    params = func["parameters"]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert params["properties"]["query"]["type"] == "string"
    assert "search query" in params["properties"]["query"]["description"].lower()

    assert "max_results" in params["properties"]
    assert params["properties"]["max_results"]["type"] == "integer"
    assert "maximum" in params["properties"]["max_results"]["description"].lower()

    # query is required, max_results has a default so it's not required
    assert "query" in params["required"]
    assert "max_results" not in params["required"]
