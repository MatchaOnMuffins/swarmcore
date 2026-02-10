# SwarmCore

Compose AI agents into sequential and parallel flows with automatic context management that stays lean as the number of agents grow.

<p align="center">
  <img src="assets/core_flow.gif" alt="SwarmCore flow demo" />
</p>

```bash
pip install swarmcore
```

## What it produces

Same model. Same prompt. The difference is coordination.

> *Evaluate the opportunity for launching an AI-powered personal nutrition coach app that uses computer vision to analyze meals and provides real-time dietary recommendations. Target market: health-conscious millennials in the US.*

**Single agent** — generic strategic assessment

<details>
<summary>Read full output</summary>

The opportunity for an AI-powered personal nutrition coach targeting health-conscious US millennials is compelling, driven by high smartphone usage, strong interest in personalized wellness, and frustration with manual calorie tracking. Computer vision–based meal analysis directly addresses a major pain point by reducing friction, while real-time dietary feedback aligns well with millennials' preference for on-demand, data-driven guidance integrated into daily life.

That said, the market is competitive and execution-sensitive: differentiation will require demonstrably accurate food recognition, culturally diverse meal coverage, and actionable recommendations that go beyond basic calorie counts. Trust will be critical, making privacy safeguards, transparent AI limitations, and careful positioning to avoid medical claims essential, alongside a clear monetization strategy (e.g., premium personalization or partnerships with fitness and health platforms).

</details>

**SwarmCore (5 agents)** — specific market data, unit economics, go-to-market strategy

<details>
<summary>Read full output</summary>

The opportunity is meaningful if we position this as *frictionless meal logging plus actionable micro-coaching*, not "perfect" automated nutrition. The US mHealth apps market is estimated at **~$12.75B (2024)**, and meal occasions are increasingly "trackable" (Circana reports **86% of eating occasions are sourced from home**), creating high usage frequency for a camera-first workflow. Strategically, we should **launch with a hybrid CV + user-confirmation experience** (detect items, then prompt 1–2 quick portion inputs) and focus differentiation on (1) speed/low-friction capture, (2) trustworthy, goal-based recommendations, and (3) privacy-first handling of images. MVP is technically feasible in **~4–6 months** using pretrained vision + a constrained food ontology + human-in-the-loop QA; true defensibility likely requires **9–18 months** of data collection, model tuning, and nutrition governance.

Key risks are (a) **portion estimation accuracy** (single-photo volume inference is unreliable), (b) **trust/privacy** (food photos can reveal sensitive context), and (c) **regulatory/claims creep**—we must stay clearly on the "wellness" side of FDA medical-device boundaries. Financially, a base-case model supports attractive unit economics: by **Year 3 ~32k paying subscribers** at **$17.99/mo** can drive **~$5–6M revenue**, with **~74% gross margin**, **blended CAC ~ $52**, and **LTV:CAC ~4.1x**. Estimated **break-even ~month 20–22** with **~$2.6M** upfront/early operating investment, contingent on holding **paid churn ≲6%** and maintaining recommendation quality to prevent post-novelty attrition.

</details>

## How it works

Agents in a flow share context automatically. SwarmCore keeps it lean — each agent sees only what it needs:

- **Immediately preceding step** → full output
- **Everything earlier** → summaries only
- **Need more?** → agents call `expand_context` at runtime to pull any prior agent's full output

```
agent_1 >> agent_2 >> ... >> agent_10

agent_10 sees: agents 1-8 [SUMMARIES] + agent_9 [FULL]
               (can expand any earlier agent on demand)
```

Summaries are extracted from `<summary>` tags in agent output. If omitted, the full output is used instead.

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
