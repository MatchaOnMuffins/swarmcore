"""
Tiered context demo — shows exactly what each agent sees, including
dynamic expansion via the ``expand_context`` tool.

    researcher >> analyst >> critic >> writer

Step 1 (researcher): no prior context
Step 2 (analyst):    researcher [FULL] (immediately preceding)
Step 3 (critic):     researcher [SUMMARY] + analyst [FULL]
                     expand_context tool available for researcher
Step 4 (writer):     researcher [SUMMARY] + analyst [SUMMARY] + critic [FULL]
                     expand_context tool available for researcher, analyst

If the LLM calls expand_context("researcher"), it gets the full
original output back as a tool result — no code changes needed.

Run:  python examples/tiered_context.py
Requires OPENAI_API_KEY (uses gpt-4o-mini by default).
"""

from __future__ import annotations

import asyncio
import textwrap
from typing import Any

import litellm

from swarmcore import Agent, Swarm

# ── Agents ────────────────────────────────────────────────────────

MODEL = "openai/gpt-4o-mini"

researcher = Agent(
    name="researcher",
    instructions=(
        "You are a research specialist. Write 2-3 detailed paragraphs "
        "about the given topic with specific data points and statistics."
    ),
    model=MODEL,
)

analyst = Agent(
    name="analyst",
    instructions=(
        "You are a data analyst. Review the research in context and "
        "identify the 3 most important insights. Explain each briefly."
    ),
    model=MODEL,
)

critic = Agent(
    name="critic",
    instructions=(
        "You are a critical reviewer. Evaluate the analysis in context "
        "for gaps, biases, or missing perspectives. Be constructive."
    ),
    model=MODEL,
)

writer = Agent(
    name="writer",
    instructions=(
        "You are a technical writer. Using the context provided, draft "
        "a polished 2-paragraph executive briefing. You need specific "
        "data points from the original research — use the expand_context "
        "tool to retrieve the researcher's full output if the summary "
        "doesn't have enough detail."
    ),
    model=MODEL,
)

agents = [researcher, analyst, critic, writer]

# ── Intercept LLM calls to capture inputs & raw outputs ──────────

_captures: list[dict[str, Any]] = []
_original_acompletion = litellm.acompletion


async def _capturing_acompletion(**kwargs: Any) -> Any:
    """Thin wrapper that records system prompt and raw response."""
    messages = kwargs.get("messages", [])
    system_msg = next(
        (m["content"] for m in messages if m["role"] == "system"), ""
    )
    tools = kwargs.get("tools", [])
    tool_names = [t["function"]["name"] for t in tools]

    response = await _original_acompletion(**kwargs)

    content = response.choices[0].message.content or ""  # type: ignore[union-attr]
    tool_calls_made = []
    tc_list = response.choices[0].message.tool_calls  # type: ignore[union-attr]
    if tc_list:
        tool_calls_made = [tc.function.name for tc in tc_list]

    _captures.append({
        "system_prompt": system_msg,
        "raw_output": content,
        "tool_names_available": tool_names,
        "tool_calls_made": tool_calls_made,
    })
    return response


litellm.acompletion = _capturing_acompletion  # type: ignore[assignment]

# ── Helpers ───────────────────────────────────────────────────────

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


def _label(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def _indent(text: str, prefix: str = "  │ ") -> str:
    return textwrap.indent(text.strip(), prefix)


def _extract_context_block(system_prompt: str) -> str | None:
    marker = "# Context from prior agents"
    idx = system_prompt.find(marker)
    if idx == -1:
        return None
    block = system_prompt[idx + len(marker) :]
    end = block.find("# Output format")
    if end != -1:
        block = block[:end]
    return block.strip()


# ── Main ──────────────────────────────────────────────────────────

flow = researcher >> analyst >> critic >> writer
swarm = Swarm(flow=flow)


async def main() -> None:
    task = "What are the key trends in AI agents in 2025?"

    print(f"\n{BOLD}Task:{RESET} {task}")
    print(f"{DIM}Flow: researcher >> analyst >> critic >> writer{RESET}")

    result = await swarm.run(task)

    # Group captures by agent (an agent with tool calls has multiple captures)
    agent_captures: dict[str, list[dict[str, Any]]] = {}
    cap_idx = 0
    for agent_result in result.history:
        name = agent_result.agent_name
        agent_captures[name] = []
        # Each LLM call produces one capture
        for _ in range(agent_result.llm_call_count):
            if cap_idx < len(_captures):
                agent_captures[name].append(_captures[cap_idx])
                cap_idx += 1

    for i, agent_result in enumerate(result.history):
        name = agent_result.agent_name
        caps = agent_captures.get(name, [])
        first_cap = caps[0] if caps else None

        _section(f"Step {i + 1}: {name}")

        # Context injected
        if first_cap:
            ctx_block = _extract_context_block(first_cap["system_prompt"])
            print(f"\n  {_label('Context injected into system prompt:')}")
            if ctx_block is None:
                print(f"  │ {DIM}(none — first agent){RESET}")
            else:
                for line in ctx_block.split("\n"):
                    if line.startswith("## "):
                        if "(summary)" in line:
                            tag = f"{YELLOW}[SUMMARY]{RESET}"
                        else:
                            tag = f"{GREEN}[FULL]{RESET}"
                        print(f"  │ {CYAN}{line}{RESET}  {tag}")
                    else:
                        print(f"  │ {DIM}{line}{RESET}")

            # Tools available
            tool_names = first_cap["tool_names_available"]
            if tool_names:
                print(
                    f"\n  {_label('Tools available:')} "
                    f"{', '.join(tool_names)}"
                )

        # Tool calls made
        if agent_result.tool_call_count > 0:
            print(f"\n  {_label('Tool calls made:')}")
            for tc in agent_result.tool_calls:
                print(
                    f"  │ {RED}{tc.tool_name}{RESET}"
                    f"({', '.join(f'{k}={v!r}' for k, v in tc.arguments.items())})"
                )
                preview = tc.result[:120] + "..." if len(tc.result) > 120 else tc.result
                print(f"  │ → {DIM}{preview}{RESET}")

        # Raw LLM response (first call only for brevity)
        if first_cap and first_cap["raw_output"]:
            raw = first_cap["raw_output"]
            print(f"\n  {_label('Raw LLM response:')}")
            if "<summary>" in raw:
                highlighted = raw.replace(
                    "<summary>", f"{YELLOW}<summary>{RESET}"
                ).replace("</summary>", f"{YELLOW}</summary>{RESET}")
                print(_indent(highlighted))
            else:
                print(_indent(raw))

        # Parsed result
        print(f"\n  {_label('Parsed summary:')}")
        print(_indent(agent_result.summary, "  │ "))
        print(f"\n  {_label('Parsed detail (stored as output):')}")
        print(_indent(agent_result.output, "  │ "))

    # Final summary
    _section("Result")
    print(f"\n  {_label('SwarmResult.output:')}")
    print(_indent(result.output))
    print()
    for r in result.history:
        print(
            f"  {r.agent_name:12s}  "
            f"output={len(r.output):>5d} chars  "
            f"summary={len(r.summary):>4d} chars  "
            f"tools={r.tool_call_count}  "
            f"tokens={r.token_usage.total_tokens:>5d}  "
            f"time={r.duration_seconds}s"
        )
    print(
        f"\n  {'total':12s}  "
        f"tokens={result.total_token_usage.total_tokens:>5d}  "
        f"time={result.duration_seconds}s\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
