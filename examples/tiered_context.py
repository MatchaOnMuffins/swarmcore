"""
    market_researcher >> tech_analyst >> financial_modeler
        >> risk_assessor >> strategist >> exec_briefer

- market_researcher:  consumer trends, TAM, competitor landscape
- tech_analyst:       technical feasibility, architecture, IP concerns
- financial_modeler:  revenue projections, unit economics, margins
- risk_assessor:      regulatory, market, and execution risks
- strategist:         go-to-market strategy (needs market + financial, not all)
- exec_briefer:       2-paragraph briefing (must be selective — can't use all 5)

Run:  python examples/tiered_context_product.py
Requires OPENAI_API_KEY (uses gpt-4o-mini by default).
"""

from __future__ import annotations

import asyncio
import textwrap
from typing import Any

import litellm

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

# ── Colors ────────────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}{RESET}")


def _indent(text: str, prefix: str = "  │ ") -> str:
    return textwrap.indent(text.strip(), prefix)


# ── Intercept LLM calls to display context hints in real time ─────

_current_agent: str | None = None
_context_shown: set[str] = set()
_original_acompletion = litellm.acompletion


def _show_hint_from_prompt(system_prompt: str, tool_names: list[str]) -> None:
    """Parse and display the context hint from the system prompt."""
    marker = "# Available context"
    idx = system_prompt.find(marker)

    print(f"  {BOLD}Context hint:{RESET}")
    if idx == -1:
        print(f"  │ {DIM}(none — first agent){RESET}")
    else:
        block = system_prompt[idx + len(marker) :]
        end = block.find("# Output format")
        if end != -1:
            block = block[:end]
        for line in block.strip().split("\n"):
            line = line.strip()
            if line.startswith("- **"):
                print(f"  │ {CYAN}{line}{RESET}")

    if tool_names:
        print(f"  {BOLD}Tools:{RESET} {', '.join(tool_names)}")


async def _streaming_acompletion(**kwargs: Any) -> Any:
    """Wrapper: display context hint on the first LLM call per agent."""
    agent = _current_agent
    if agent and agent not in _context_shown:
        _context_shown.add(agent)
        messages = kwargs.get("messages", [])
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        tools: list[dict[str, Any]] = kwargs.get("tools", [])
        tool_names = [t["function"]["name"] for t in tools]
        _show_hint_from_prompt(system_msg, tool_names)

    return await _original_acompletion(**kwargs)


litellm.acompletion = _streaming_acompletion  # type: ignore[assignment]


# ── Hook handler for streaming lifecycle events ───────────────────


def _handle_event(event: Event) -> None:
    global _current_agent
    data = event.data

    if event.type == EventType.STEP_START:
        step_idx = data["step_index"]
        agents = data["agents"]
        parallel = data.get("parallel", False)
        label = " | ".join(agents) if parallel else agents[0]
        kind = " (parallel)" if parallel else ""
        _section(f"Step {step_idx + 1}: {label}{kind}")

    elif event.type == EventType.AGENT_START:
        _current_agent = data["agent"]
        print(f"\n  {BOLD}▶ {data['agent']}{RESET}")

    elif event.type == EventType.LLM_CALL_START:
        idx = data["call_index"]
        print(f"  {DIM}LLM call {idx}...{RESET}", end="", flush=True)

    elif event.type == EventType.LLM_CALL_END:
        duration = data["duration_seconds"]
        tokens = data["total_tokens"]
        finish = data.get("finish_reason", "")
        suffix = f" → {YELLOW}tool_calls{RESET}" if finish == "tool_calls" else ""
        print(f" {duration}s, {tokens} tok{suffix}")

    elif event.type == EventType.TOOL_CALL_START:
        tool = data["tool"]
        args = data.get("arguments", {})
        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
        print(f"  │ {RED}⚡ {tool}({args_str}){RESET}")

    elif event.type == EventType.TOOL_CALL_END:
        duration = data["duration_seconds"]
        print(f"  │   {DIM}→ returned ({duration}s){RESET}")

    elif event.type == EventType.AGENT_END:
        _current_agent = None
        print(f"  {GREEN}✓ {data['agent']} done ({data['duration_seconds']}s){RESET}")

    elif event.type == EventType.AGENT_ERROR:
        print(f"  {RED}✗ {data.get('agent', '?')} error: {data.get('error')}{RESET}")


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


async def main() -> None:
    task = (
        "Evaluate the opportunity for launching an AI-powered personal "
        "nutrition coach app that uses computer vision to analyze meals "
        "and provides real-time dietary recommendations. Target market: "
        "health-conscious millennials in the US."
    )

    agent_names = [
        "market_researcher",
        "tech_analyst",
        "financial_modeler",
        "risk_assessor",
        "strategist",
        "exec_briefer",
    ]

    print(f"\n{BOLD}Task:{RESET} {task}")
    print(f"{DIM}Flow: {' >> '.join(agent_names)} (pull mode){RESET}")

    result = await swarm.run(task)

    # ── Agent outputs ─────────────────────────────────────────────
    _section("Agent Outputs")
    for r in result.history:
        print(f"\n  {BOLD}{r.agent_name}{RESET}")
        if r.summary and r.summary != r.output:
            print(f"  {YELLOW}Summary:{RESET} {DIM}(from <summary> tags){RESET}")
            print(_indent(r.summary))
        else:
            print(f"  {DIM}(no <summary> tags — full output used as summary){RESET}")
        output = r.output.strip()
        if len(output) > 600:
            output = output[:600] + f"\n{DIM}... ({len(r.output)} chars total){RESET}"
        print(f"  {GREEN}Output:{RESET}")
        print(_indent(output))

    # ── Context pull analysis ─────────────────────────────────────
    _section("Context Pull Analysis")
    print()
    print(
        "  This section shows which agents pulled full context from which prior agents."
    )
    print(
        f"  {BOLD}Ideal behavior:{RESET} later agents should be selective, "
        f"not pull everything.\n"
    )
    for r in result.history:
        if r.tool_call_count > 0:
            pulls = []
            for tc in r.tool_calls:
                if tc.tool_name == "get_context":
                    agent_arg = tc.arguments.get("agent_name", "?")
                    pulls.append(agent_arg)

            available = [
                h.agent_name
                for h in result.history
                if h.agent_name != r.agent_name
                and result.history.index(h) < result.history.index(r)
            ]
            n_available = len(available)
            n_pulled = len(pulls)

            selectivity = "SELECTIVE ✓" if n_pulled < n_available else "PULLED ALL ⚠"

            print(
                f"  {r.agent_name:18s}  available={n_available}  pulled={n_pulled}  {selectivity}"
            )
            if pulls:
                print(f"  {'':18s}  → {', '.join(pulls)}")
            if available and n_pulled < n_available:
                skipped = [a for a in available if a not in pulls]
                print(f"  {'':18s}  {DIM}skipped: {', '.join(skipped)}{RESET}")
    print()

    # ── Token usage ───────────────────────────────────────────────
    _section("Token Usage")
    print()
    for r in result.history:
        print(
            f"  {r.agent_name:18s}  "
            f"tokens={r.token_usage.total_tokens:>5d}  "
            f"calls={r.llm_call_count}  "
            f"tools={r.tool_call_count}  "
            f"time={r.duration_seconds}s"
        )
    print(
        f"\n  {'total':18s}  "
        f"tokens={result.total_token_usage.total_tokens:>5d}  "
        f"time={result.duration_seconds}s"
    )

    # ── Final output ──────────────────────────────────────────────
    _section("Final Output")
    print()
    print(_indent(result.output.strip(), "  "))
    print()


if __name__ == "__main__":
    asyncio.run(main())
