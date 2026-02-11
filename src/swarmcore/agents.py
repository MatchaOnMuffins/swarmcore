"""Pre-built agent factories for common workflow roles."""

from __future__ import annotations

from typing import Any, Callable

from swarmcore.agent import Agent
from swarmcore.tools import search_web

_UNSET: Any = object()

_RESEARCHER_INSTRUCTIONS = (
    "You are a research specialist. Your job is to gather comprehensive, "
    "accurate information on the given topic. Use your search tools to find "
    "relevant data, statistics, and sources. Cross-reference multiple sources "
    "to verify claims. Present your findings in a structured format with clear "
    "sections. Always cite your sources with URLs when available. Focus on "
    "recency and relevance — prefer authoritative, up-to-date sources over "
    "older or less credible ones."
)

_ANALYST_INSTRUCTIONS = (
    "You are an analytical specialist. Your job is to examine the information "
    "provided and extract meaningful insights. Identify key patterns, trends, "
    "and relationships in the data. Evaluate strengths, weaknesses, "
    "opportunities, and risks. Support your conclusions with specific evidence "
    "from the context. Structure your analysis with clear headings and use "
    "bullet points for key findings. Distinguish between facts, inferences, "
    "and opinions."
)

_WRITER_INSTRUCTIONS = (
    "You are a writing specialist. Your job is to produce clear, well-structured "
    "content based on the information and analysis provided in context. Organize "
    "your writing with a logical flow — introduction, body, and conclusion. Use "
    "precise language and vary sentence structure for readability. Adapt your "
    "tone to match the subject matter. Include specific details and examples to "
    "support key points. Ensure smooth transitions between sections."
)

_EDITOR_INSTRUCTIONS = (
    "You are an editorial specialist. Your job is to synthesize and polish "
    "content from multiple sources into a cohesive final output. Resolve any "
    "contradictions or redundancies between inputs. Improve clarity, flow, and "
    "consistency of tone throughout. Tighten prose by removing filler and "
    "redundancy while preserving key information. Ensure the final piece reads "
    "as a unified document, not a patchwork of separate contributions. Check "
    "for logical coherence and factual consistency."
)

_SUMMARIZER_INSTRUCTIONS = (
    "You are a summarization specialist. Your job is to distill the provided "
    "content into a concise, executive-level brief. Capture the most critical "
    "points, key findings, and actionable recommendations. Use bullet points "
    "for scanability. Lead with the single most important takeaway. Omit "
    "background context that the reader already knows. Keep the summary to "
    "roughly 20% of the original length while retaining all essential "
    "information. End with clear next steps or recommendations if applicable."
)


def _build_agent(
    *,
    default_name: str,
    default_instructions: str,
    default_tools: list[Callable[..., Any]] | None,
    name: str | None,
    instructions: str | None,
    model: str,
    tools: Any,
    timeout: float | None,
    max_retries: int | None,
    max_turns: int | None,
) -> Agent:
    """Shared builder for all factory functions."""
    effective_name = name if name is not None else default_name
    effective_instructions = (
        instructions if instructions is not None else default_instructions
    )

    if tools is _UNSET:
        effective_tools = default_tools
    else:
        effective_tools = tools if tools else None

    return Agent(
        name=effective_name,
        instructions=effective_instructions,
        model=model,
        tools=effective_tools,
        timeout=timeout,
        max_retries=max_retries,
        max_turns=max_turns,
    )


def researcher(
    *,
    name: str = "researcher",
    instructions: str | None = None,
    model: str = "anthropic/claude-opus-4-6",
    tools: list[Callable[..., Any]] | None = _UNSET,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_turns: int | None = None,
) -> Agent:
    """Create a research agent that gathers information and finds data/sources.

    By default includes ``search_web`` as a tool. Pass ``tools=[]`` or
    ``tools=None`` to disable, or provide your own tool list.
    """
    return _build_agent(
        default_name="researcher",
        default_instructions=_RESEARCHER_INSTRUCTIONS,
        default_tools=[search_web],
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        timeout=timeout,
        max_retries=max_retries,
        max_turns=max_turns,
    )


def analyst(
    *,
    name: str = "analyst",
    instructions: str | None = None,
    model: str = "anthropic/claude-opus-4-6",
    tools: list[Callable[..., Any]] | None = _UNSET,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_turns: int | None = None,
) -> Agent:
    """Create an analysis agent that examines data and identifies insights/trends."""
    return _build_agent(
        default_name="analyst",
        default_instructions=_ANALYST_INSTRUCTIONS,
        default_tools=None,
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        timeout=timeout,
        max_retries=max_retries,
        max_turns=max_turns,
    )


def writer(
    *,
    name: str = "writer",
    instructions: str | None = None,
    model: str = "anthropic/claude-opus-4-6",
    tools: list[Callable[..., Any]] | None = _UNSET,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_turns: int | None = None,
) -> Agent:
    """Create a writing agent that drafts structured content from context."""
    return _build_agent(
        default_name="writer",
        default_instructions=_WRITER_INSTRUCTIONS,
        default_tools=None,
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        timeout=timeout,
        max_retries=max_retries,
        max_turns=max_turns,
    )


def editor(
    *,
    name: str = "editor",
    instructions: str | None = None,
    model: str = "anthropic/claude-opus-4-6",
    tools: list[Callable[..., Any]] | None = _UNSET,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_turns: int | None = None,
) -> Agent:
    """Create an editorial agent that polishes and synthesizes multiple inputs."""
    return _build_agent(
        default_name="editor",
        default_instructions=_EDITOR_INSTRUCTIONS,
        default_tools=None,
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        timeout=timeout,
        max_retries=max_retries,
        max_turns=max_turns,
    )


def summarizer(
    *,
    name: str = "summarizer",
    instructions: str | None = None,
    model: str = "anthropic/claude-opus-4-6",
    tools: list[Callable[..., Any]] | None = _UNSET,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_turns: int | None = None,
) -> Agent:
    """Create a summarization agent that condenses content into an executive-level brief."""
    return _build_agent(
        default_name="summarizer",
        default_instructions=_SUMMARIZER_INSTRUCTIONS,
        default_tools=None,
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        timeout=timeout,
        max_retries=max_retries,
        max_turns=max_turns,
    )
