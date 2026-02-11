from __future__ import annotations


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted results.

    query: The search query string
    max_results: Maximum number of results to return (default 5)
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Error: ddgs is not installed. Install it with: pip install ddgs"

    results = DDGS().text(query, max_results=max_results)

    if not results:
        return "No results found."

    formatted: list[str] = []
    for r in results:
        formatted.append(f"**{r['title']}**\n{r['body']}\n{r['href']}")

    return "\n\n".join(formatted)
