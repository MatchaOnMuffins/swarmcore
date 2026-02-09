"""
Research team example: a multi-agent workflow.

    researcher >> [analyst, writer] >> editor

Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio

from swarmcore import Agent, Swarm

researcher = Agent(
    name="researcher",
    instructions="You are a research specialist. Gather key facts and data about the given topic. Write detailed notes.",
    model="anthropic/claude-opus-4-6",
)

analyst = Agent(
    name="analyst",
    instructions="You are a data analyst. Review the research in context and identify the most important insights and trends.",
    model="anthropic/claude-opus-4-6",
)

writer = Agent(
    name="writer",
    instructions="You are a technical writer. Using the research in context, draft a clear summary report.",
    model="anthropic/claude-opus-4-6",
)

editor = Agent(
    name="editor",
    instructions="You are an editor. Combine the analysis and draft from context into a polished final report.",
    model="anthropic/claude-opus-4-6",
)

swarm = Swarm(flow=researcher >> (analyst | writer) >> editor)


async def main() -> None:
    result = await swarm.run("What are the key trends in AI agents in 2025?")
    print(result.output)
    print("\n--- Context Keys ---")
    for name in result.context:
        print(f"  {name}: {len(result.context[name])} chars")
    print(f"\n--- {len(result.history)} agents ran ---")
    for r in result.history:
        print(f"  {r.agent_name}: {r.duration_seconds}s, {r.token_usage.total_tokens} tokens")


if __name__ == "__main__":
    asyncio.run(main())
