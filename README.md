# SwarmCore

Coordinate AI agents in a workflow. `pip install swarmcore`.

<p align="center">
  <img src="assets/core_flow.gif" alt="SwarmCore flow demo" />
</p>

## Install

```bash
pip install swarmcore
```

## Quickstart

```python
import asyncio
from swarmcore import Agent, Swarm

researcher = Agent(
    name="researcher",
    instructions="Find key information about the given topic. Write detailed notes.",
    model="anthropic/claude-opus-4-6",
)

writer = Agent(
    name="writer",
    instructions="Using the research notes in context, write a clear summary report.",
    model="anthropic/claude-opus-4-6",
)

swarm = Swarm(flow=researcher >> writer)

result = asyncio.run(swarm.run("What are the key trends in AI agents in 2025?"))
print(result.output)
```

The researcher runs first. Its output is stored in shared context. The writer runs next and sees the research notes in its prompt. `result.output` is the writer's final response.

## Flows

Compose agents with `>>` (sequential) and `|` (parallel):

```python
# Sequential: planner runs, then writer
swarm = Swarm(flow=planner >> writer)

# Parallel: researcher and critic run concurrently
swarm = Swarm(flow=(researcher | critic) >> writer)

# Mixed: planner, then researcher+critic in parallel, then writer
swarm = Swarm(flow=planner >> (researcher | critic) >> writer)
```

Or use the functional API:

```python
from swarmcore import chain, parallel

swarm = Swarm(flow=chain(planner, parallel(researcher, critic), writer))
```

All parallel agents see the same context snapshot from before their step. They don't see each other's outputs during execution.

## Tools

Give agents tools as plain Python functions:

```python
def search_web(query: str) -> str:
    """Search the web for information."""
    # your implementation
    return results

agent = Agent(
    name="researcher",
    instructions="Use the search tool to find information.",
    model="anthropic/claude-opus-4-6",
    tools=[search_web],
)
```

Functions are automatically converted to OpenAI function-calling schemas using type hints and docstrings. The agent will call tools in a loop until it produces a final text response.

Async functions work too:

```python
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    ...
```

## Tiered Context

As agents chain together, every prior agent's full output would normally be injected into downstream system prompts, growing linearly. SwarmCore solves this automatically:

- Each agent produces a **summary** (via `<summary>` tags) and a **full output**
- Downstream agents see **full output** from the immediately preceding step and **summaries** from everything earlier
- When a summary isn't enough, agents can call the **`expand_context` tool** to retrieve any prior agent's full output on demand

```
researcher >> analyst >> critic >> writer

analyst sees:  researcher [FULL]
critic sees:   researcher [SUMMARY] + analyst [FULL]
writer sees:   researcher [SUMMARY] + analyst [SUMMARY] + critic [FULL]
               (can call expand_context("researcher") to get full output)
```

The `expand_context` tool is injected automatically whenever summarized context exists. Agents decide at runtime whether a summary is enough or if they need the full thing — just mention the tool in the agent's instructions:

```python
writer = Agent(
    name="writer",
    instructions=(
        "Write an executive briefing using the context provided. "
        "If you need specific data points from earlier research, "
        "use the expand_context tool to get the full output."
    ),
    model="anthropic/claude-opus-4-6",
)

swarm = Swarm(flow=researcher >> analyst >> critic >> writer)
result = asyncio.run(swarm.run("AI trends in 2025"))

# Each result carries both the full output and its summary
for r in result.history:
    print(f"{r.agent_name}: {len(r.output)} chars, summary: {len(r.summary)} chars")
```

No configuration needed. If an agent's response doesn't include `<summary>` tags, the full output is used everywhere (graceful degradation).

## Models

SwarmCore uses [LiteLLM](https://docs.litellm.ai/) under the hood. Any LiteLLM-compatible model string works:

```python
Agent(name="a", instructions="...", model="anthropic/claude-opus-4-6")
Agent(name="b", instructions="...", model="openai/gpt-4o")
Agent(name="c", instructions="...", model="ollama/llama3")
Agent(name="d", instructions="...", model="groq/llama-3.1-8b-instant")
```

Set the appropriate API key for your provider (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc).

## API Reference

### `Agent(name, instructions, model, tools)`

| Param | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique identifier used in context keys |
| `instructions` | `str` | required | System prompt for this agent |
| `model` | `str` | `"anthropic/claude-opus-4-6"` | LiteLLM model string |
| `tools` | `list[Callable]` | `None` | Python functions for tool calling |

Supports `>>` (sequential) and `|` (parallel) operators for composing flows.

### `Swarm(flow)`

| Param | Type | Description |
|---|---|---|
| `flow` | `Flow` | Execution plan built via operators or `chain()`/`parallel()` |

### `chain(*agents_or_groups)` / `parallel(*agents)`

```python
from swarmcore import chain, parallel

# Functional flow construction
flow = chain(a, parallel(b, c), d)
swarm = Swarm(flow=flow)
```

`parallel()` requires at least 2 agents.

### `SwarmResult`

| Field | Type | Description |
|---|---|---|
| `output` | `str` | Final agent's output |
| `context` | `dict[str, str]` | All agent outputs keyed by agent name |
| `history` | `list[AgentResult]` | Ordered execution results |

### `AgentResult`

| Field | Type | Description |
|---|---|---|
| `agent_name` | `str` | Agent that produced this result |
| `input_task` | `str` | The task string passed to the agent |
| `output` | `str` | Agent's text output (summary tags stripped) |
| `summary` | `str` | Short summary extracted from `<summary>` tags, or full output if none |
| `model` | `str` | Model used |
| `duration_seconds` | `float` | Execution time |
| `token_usage` | `TokenUsage` | Token counts (prompt, completion, total) |

## Roadmap

- Conditional branching in flows
- Streaming responses
- Retry policies
- Agent memory / state persistence
- Dynamic re-planning

## License

MIT
