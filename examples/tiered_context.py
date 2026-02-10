"""
Six-agent product-evaluation pipeline demonstrating pull-mode context.

    (market_researcher | tech_analyst | financial_modeler)
        >> risk_assessor >> strategist >> exec_briefer

Run:  python examples/tiered_context.py
"""

from __future__ import annotations

import asyncio
import textwrap

from ddgs import DDGS

from swarmcore import Agent, Event, EventType, Hooks, Swarm

# ── Tools ─────────────────────────────────────────────────────────


def search_web(query: str) -> str:
    """Search the web for current information.

    query: The search query string
    """
    results = DDGS().text(query, max_results=5)
    return "\n\n".join(f"**{r['title']}**\n{r['body']}\n{r['href']}" for r in results)


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

# ── Formatting helpers ────────────────────────────────────────────

G, Y, R, DIM, BOLD, RST = (
    "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"
)


def _section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 70}\n  {title}\n{'─' * 70}{RST}")


def _indent(text: str, prefix: str = "  │ ") -> str:
    return textwrap.indent(text.strip(), prefix)


# ── Hook handler ─────────────────────────────────────────────────


def _handle_event(event: Event) -> None:
    d = event.data
    match event.type:
        case EventType.STEP_START:
            agents = d["agents"]
            par = d.get("parallel", False)
            label = (" | ".join(agents) if par else agents[0])
            _section(f"Step {d['step_index'] + 1}: {label}{' (parallel)' if par else ''}")
        case EventType.AGENT_START:
            print(f"\n  {BOLD}▶ {d['agent']}{RST}")
        case EventType.LLM_CALL_START:
            print(f"  {DIM}LLM call {d['call_index']}...{RST}", end="", flush=True)
        case EventType.LLM_CALL_END:
            tc = f" → {Y}tool_calls{RST}" if d.get("finish_reason") == "tool_calls" else ""
            print(f" {d['duration_seconds']}s, {d['total_tokens']} tok{tc}")
        case EventType.TOOL_CALL_START:
            args = ", ".join(f'{k}="{v}"' for k, v in d.get("arguments", {}).items())
            print(f"  │ {R}⚡ {d['tool']}({args}){RST}")
        case EventType.TOOL_CALL_END:
            print(f"  │   {DIM}→ returned ({d['duration_seconds']}s){RST}")
        case EventType.AGENT_END:
            print(f"  {G}✓ {d['agent']} done ({d['duration_seconds']}s){RST}")
        case EventType.AGENT_ERROR:
            print(f"  {R}✗ {d.get('agent', '?')} error: {d.get('error')}{RST}")


# ── Main ──────────────────────────────────────────────────────────

flow = (
    (market_researcher | tech_analyst | financial_modeler)
    >> risk_assessor
    >> strategist
    >> exec_briefer
)

hooks = Hooks()
hooks.on_all(_handle_event)
swarm = Swarm(flow=flow, hooks=hooks, context_mode="pull")

TASK = (
    "Evaluate the opportunity for launching an AI-powered personal "
    "nutrition coach app that uses computer vision to analyze meals "
    "and provides real-time dietary recommendations. Target market: "
    "health-conscious millennials in the US."
)


async def main() -> None:
    print(f"\n{BOLD}Task:{RST} {TASK}\n")
    result = await swarm.run(TASK)

    # ── Context pull analysis ─────────────────────────────────────
    _section("Context Pull Analysis")
    for r in result.history:
        pulls = [tc.arguments.get("agent_name", "?")
                 for tc in r.tool_calls if tc.tool_name == "get_context"]
        if not pulls:
            continue
        prior = [h.agent_name for h in result.history
                 if h.agent_name != r.agent_name
                 and result.history.index(h) < result.history.index(r)]
        tag = "SELECTIVE ✓" if len(pulls) < len(prior) else "PULLED ALL ⚠"
        skipped = [a for a in prior if a not in pulls]
        print(f"  {r.agent_name:18s}  {len(pulls)}/{len(prior)} pulled  {tag}")
        if skipped:
            print(f"  {'':18s}  {DIM}skipped: {', '.join(skipped)}{RST}")

    # ── Token usage ───────────────────────────────────────────────
    _section("Token Usage")
    for r in result.history:
        print(f"  {r.agent_name:18s}  {r.token_usage.total_tokens:>5d} tok  "
              f"{r.llm_call_count} calls  {r.tool_call_count} tools  {r.duration_seconds}s")
    print(f"\n  {'TOTAL':18s}  {result.total_token_usage.total_tokens:>5d} tok  "
          f"{result.duration_seconds}s")

    # ── Final output ──────────────────────────────────────────────
    _section("Final Output")
    print(_indent(result.output.strip(), "  "))
    print()


if __name__ == "__main__":
    asyncio.run(main())
