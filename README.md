# SwarmCore

Coordinate AI agents in a workflow. `pip install swarmcore`.

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

swarm = Swarm(
    agents=[researcher, writer],
    flow="researcher >> writer",
)

result = asyncio.run(swarm.run("What are the key trends in AI agents in 2025?"))
print(result.output)
```

The researcher runs first. Its output is stored in shared context. The writer runs next and sees the research notes in its prompt. `result.output` is the writer's final response.

## Parallel Flows

Fan out to multiple agents, then fan back in:

```python
swarm = Swarm(
    agents=[planner, researcher, critic, writer],
    flow="planner >> [researcher, critic] >> writer",
)
```

`planner` runs first. Then `researcher` and `critic` run **in parallel** (concurrent LLM calls). Then `writer` runs with all three outputs in context.

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
| `name` | `str` | required | Unique identifier used in flow strings and context keys |
| `instructions` | `str` | required | System prompt for this agent |
| `model` | `str` | `"anthropic/claude-opus-4-6"` | LiteLLM model string |
| `tools` | `list[Callable]` | `None` | Python functions for tool calling |

### `Swarm(agents, flow)`

| Param | Type | Description |
|---|---|---|
| `agents` | `list[Agent]` | All agents referenced in the flow |
| `flow` | `str` | Execution flow: `"a >> b"` (sequential), `"a >> [b, c] >> d"` (parallel) |

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
| `output` | `str` | Agent's text output |
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
