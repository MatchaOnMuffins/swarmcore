# SwarmCore

What is Swarm Core?

Scalable AI agent coordination. Compose agents into sequential and parallel flows with automatic context management that stays lean as the number of agents grow.

TL;DR: Multi-agent orchestration with selective context expansion.

<p align="center">
  <img src="assets/core_flow.gif" alt="SwarmCore flow demo" />
</p>

```bash
pip install swarmcore
```

## Quickstart

```python
import asyncio
from swarmcore import Agent, Swarm

researcher = Agent(name="researcher", instructions="Research the topic.", model="anthropic/claude-sonnet-4-20250514")
writer = Agent(name="writer", instructions="Write a report from the research.", model="anthropic/claude-sonnet-4-20250514")

result = asyncio.run(Swarm(flow=researcher >> writer).run("AI agent trends in 2025"))
print(result.output)
```

The researcher runs first, its output goes into shared context, and the writer sees it automatically.

## Tiered Context

Most multi-agent frameworks dump every prior agent's full output into the next prompt. That blows up at scale. SwarmCore keeps it tight:

- **Immediately preceding step** → full output
- **Everything earlier** → summaries only
- **Need more?** → agents call `expand_context` at runtime to pull any prior agent's full output

```
agent_1 >> agent_2 >> ... >> agent_10

agent_10 sees: agents 1-8 [SUMMARIES] + agent_9 [FULL]
               (can expand any earlier agent on demand)
```

The tool and prompt hint are injected automatically. Agents produce summaries via `<summary>` tags in their output — if omitted, the full output is used instead.

## Flows

`>>` for sequential, `|` for parallel:

```python
planner >> writer                        # sequential
(researcher | critic) >> writer          # parallel then sequential
planner >> (researcher | critic) >> writer  # mixed
```

Or the functional API:

```python
from swarmcore import chain, parallel
Swarm(flow=chain(planner, parallel(researcher, critic), writer))
```

Parallel agents share the same context snapshot — they don't see each other's outputs.

## Tools

Plain Python functions:

```python
def search_web(query: str) -> str:
    """Search the web for information."""
    return results

agent = Agent(name="researcher", instructions="...", tools=[search_web])
```

Type hints and docstrings are converted to tool schemas automatically. Sync and async functions both work.

## Models

Any [LiteLLM](https://docs.litellm.ai/)-compatible model:

```python
Agent(name="a", instructions="...", model="anthropic/claude-sonnet-4-20250514")
Agent(name="b", instructions="...", model="openai/gpt-4o")
Agent(name="c", instructions="...", model="ollama/llama3")
Agent(name="d", instructions="...", model="groq/llama-3.1-8b-instant")
```

## API

### `Agent(name, instructions, model, tools=None)`

| Param | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Identifier used in context keys |
| `instructions` | `str` | required | System prompt |
| `model` | `str` | `"anthropic/claude-sonnet-4-20250514"` | LiteLLM model string |
| `tools` | `list[Callable]` | `None` | Tool functions |

### `Swarm(flow)`

| Param | Type | Description |
|---|---|---|
| `flow` | `Flow` | Execution plan from operators or `chain()`/`parallel()` |

### `SwarmResult`

| Field | Type | Description |
|---|---|---|
| `output` | `str` | Final agent's output |
| `context` | `dict[str, str]` | All outputs keyed by agent name |
| `history` | `list[AgentResult]` | Ordered execution results |

### `AgentResult`

| Field | Type | Description |
|---|---|---|
| `agent_name` | `str` | Agent that produced this result |
| `output` | `str` | Text output (summary tags stripped) |
| `summary` | `str` | Summary from `<summary>` tags, or full output |
| `model` | `str` | Model used |
| `duration_seconds` | `float` | Wall-clock time |
| `token_usage` | `TokenUsage` | Token counts |

## License

MIT
