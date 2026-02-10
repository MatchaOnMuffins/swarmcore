from __future__ import annotations

from typing import Any, Callable

from swarmcore.context import SharedContext


def make_list_context_tool(ctx: SharedContext) -> Callable[[], str]:
    """Create a ``list_context`` tool bound to the given context."""

    def list_context() -> str:
        """List all available agent outputs with summaries and sizes.

        Returns a markdown listing of agent names, their one-line summaries,
        and character counts.  Call this to discover what prior agent outputs
        are available before retrieving them.
        """
        entries = ctx.entries()
        if not entries:
            return "No agent outputs available yet."
        lines = []
        for name, summary, _full, char_count in entries:
            lines.append(f"- **{name}** ({char_count} chars): {summary}")
        return "\n".join(lines)

    return list_context


def make_get_context_tool(ctx: SharedContext) -> Callable[[str], str]:
    """Create a ``get_context`` tool bound to the given context."""

    def get_context(agent_name: str) -> str:
        """Retrieve the full output from a prior agent.

        agent_name: The name of the agent whose full output you want
        """
        full = ctx.get(agent_name)
        if full is not None:
            return full
        available = ctx.keys()
        if not available:
            return "No agent outputs available yet."
        return (
            f"No context found for agent '{agent_name}'. "
            f"Available agents: {', '.join(available)}"
        )

    return get_context


def make_search_context_tool(ctx: SharedContext) -> Callable[[str], str]:
    """Create a ``search_context`` tool bound to the given context."""

    def search_context(query: str) -> str:
        """Search across all prior agent outputs for lines matching a pattern.

        Returns matching lines grouped by agent name.  The query is a regex
        pattern, but simple keywords work too (e.g. "revenue", "TAM",
        "\\$[0-9]+" for dollar amounts).  Use "|" to combine terms:
        "TAM|market size|addressable".

        query: Regex pattern or keyword to search for (case-sensitive)
        """
        results = ctx.search(query)
        if not results:
            return f"No matches found for '{query}'."
        sections: list[str] = []
        for name, lines in results.items():
            sections.append(f"**{name}**:\n" + "\n".join(f"  {line}" for line in lines))
        return "\n\n".join(sections)

    return search_context


def make_context_tools(ctx: SharedContext) -> list[Callable[..., Any]]:
    """Create all pull-mode context tools bound to the given context."""
    return [
        make_list_context_tool(ctx),
        make_get_context_tool(ctx),
        make_search_context_tool(ctx),
    ]
