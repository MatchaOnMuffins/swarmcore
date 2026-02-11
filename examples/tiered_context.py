"""
Six-agent product-evaluation pipeline demonstrating pull-mode context.

    (market_researcher | tech_analyst | financial_modeler)
        >> risk_assessor >> strategist >> exec_briefer

Run:  python examples/tiered_context.py
"""

from __future__ import annotations

import asyncio
import textwrap

from swarmcore import Agent, Swarm, console_hooks, search_web

# ── Agents ────────────────────────────────────────────────────────

MODEL = "openai/gpt-5.2"

market_researcher = Agent(
    name="market_researcher",
    instructions=(
        "You are a market research specialist. Use search_web to find "
        "current data on the target market for the given product/initiative. "
        "Include: total addressable market (TAM) with dollar figures, "
        "consumer demographic breakdown, top 3 competitors with market "
        "share, and recent consumer behavior trends. "
        "Cite your sources with specific numbers."
    ),
    model=MODEL,
    tools=[search_web],
)

tech_analyst = Agent(
    name="tech_analyst",
    instructions=(
        "You are a technology analyst. Evaluate the technical feasibility "
        "of the given product/initiative. Cover: required tech stack and "
        "architecture, build vs. buy tradeoffs, estimated development "
        "timeline, IP/patent landscape, and key technical risks. "
        "Be specific about technologies, frameworks, and timelines."
    ),
    model=MODEL,
)

financial_modeler = Agent(
    name="financial_modeler",
    instructions=(
        "You are a financial analyst. Build a financial outlook for the "
        "given product/initiative. Include: 3-year revenue projections, "
        "unit economics (CAC, LTV, margins), break-even timeline, "
        "required investment, and key financial assumptions. "
        "Present specific dollar amounts and percentages."
    ),
    model=MODEL,
)

risk_assessor = Agent(
    name="risk_assessor",
    instructions=(
        "You are a risk analyst. Identify and evaluate risks for the "
        "given product/initiative. Categorize risks as: regulatory/legal, "
        "market/competitive, technical/execution, and financial. "
        "Rate each risk (high/medium/low), describe potential impact, "
        "and suggest mitigations. Be specific, not generic."
    ),
    model=MODEL,
)

strategist = Agent(
    name="strategist",
    instructions=(
        "You are a go-to-market strategist. Design a launch strategy for "
        "the given product/initiative. You need the market research for "
        "targeting and the financial model for pricing — retrieve those "
        "full outputs using get_context. You probably do NOT need the full "
        "technical analysis (the summary should suffice for your purposes). "
        "Cover: positioning, pricing strategy, channel mix, launch phases, "
        "and first-year milestones with specific targets."
    ),
    model=MODEL,
)

exec_briefer = Agent(
    name="exec_briefer",
    instructions=(
        "You are an executive communications specialist. Write a crisp "
        "2-paragraph executive briefing (250 words max) for the C-suite. "
        "Paragraph 1: opportunity size and strategic recommendation. "
        "Paragraph 2: key risks and financial headline numbers.\n\n"
        "IMPORTANT: You have summaries from 5 prior analysts. You do NOT "
        "need full detail from all of them. Use list_context to see what's "
        "available, then selectively retrieve only the 2-3 outputs that "
        "contain the specific data points your briefing needs. A good "
        "briefing is selective, not exhaustive."
    ),
    model=MODEL,
)

# ── Pipeline ─────────────────────────────────────────────────────

flow = (
    (market_researcher | tech_analyst | financial_modeler)
    >> risk_assessor
    >> strategist
    >> exec_briefer
)

swarm = Swarm(
    flow=flow,
    hooks=console_hooks(verbose=True),
    timeout=60.0,
    max_retries=3,
)

TASK = (
    "How have different robotics applications historically used different "
    "forms of kalman filters? What are the impacts of those different forms "
    "and when do you select them? Is it more like the complexity of your "
    "localization model, or is it based on your individual sensor inputs?"
)


async def main() -> None:
    result = await swarm.run(TASK)

    result.print_summary()

    print(textwrap.indent(result.output.strip(), "  "))


if __name__ == "__main__":
    asyncio.run(main())
