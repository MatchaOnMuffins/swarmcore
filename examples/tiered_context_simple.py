"""
Tiered context pipeline using pre-built agent factories.

    researcher >> (analyst | writer) >> editor

Run:  python examples/tiered_context_simple.py
"""

from __future__ import annotations

import asyncio
import textwrap

from swarmcore import Swarm, analyst, console_hooks, editor, researcher, writer

flow = researcher() >> (analyst() | writer()) >> editor()

swarm = Swarm(
    flow=flow, hooks=console_hooks(verbose=True), timeout=600.0, max_retries=3
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
