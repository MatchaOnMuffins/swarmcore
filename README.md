# SwarmCore

Multi-agent orchestration for Python. Compose agents into sequential and parallel flows with context sharing that scales.

<p align="center">
  <img src="assets/core_flow.gif" alt="SwarmCore flow demo" />
</p>

```bash
pip install swarmcore
```

## Output comparison

Both outputs below use the same model and prompt. The first is a single agent call; the second is a five-agent SwarmCore flow.

> *Evaluate the opportunity for launching an AI-powered personal nutrition coach app that uses computer vision to analyze meals and provides real-time dietary recommendations. Target market: health-conscious millennials in the US.*

**Single agent** — high-level strategic assessment

<details>
<summary>View output</summary>

The opportunity for an AI-powered personal nutrition coach targeting health-conscious US millennials is compelling, driven by high smartphone usage, strong interest in personalized wellness, and frustration with manual calorie tracking. Computer vision–based meal analysis directly addresses a major pain point by reducing friction, while real-time dietary feedback aligns well with millennials' preference for on-demand, data-driven guidance integrated into daily life.

That said, the market is competitive and execution-sensitive: differentiation will require demonstrably accurate food recognition, culturally diverse meal coverage, and actionable recommendations that go beyond basic calorie counts. Trust will be critical, making privacy safeguards, transparent AI limitations, and careful positioning to avoid medical claims essential, alongside a clear monetization strategy (e.g., premium personalization or partnerships with fitness and health platforms).

</details>

**Five-agent flow** — market sizing, unit economics, go-to-market strategy

<details>
<summary>View output</summary>

The opportunity is meaningful if we position this as *frictionless meal logging plus actionable micro-coaching*, not "perfect" automated nutrition. The US mHealth apps market is estimated at **~$12.75B (2024)**, and meal occasions are increasingly "trackable" (Circana reports **86% of eating occasions are sourced from home**), creating high usage frequency for a camera-first workflow. Strategically, we should **launch with a hybrid CV + user-confirmation experience** (detect items, then prompt 1–2 quick portion inputs) and focus differentiation on (1) speed/low-friction capture, (2) trustworthy, goal-based recommendations, and (3) privacy-first handling of images. MVP is technically feasible in **~4–6 months** using pretrained vision + a constrained food ontology + human-in-the-loop QA; true defensibility likely requires **9–18 months** of data collection, model tuning, and nutrition governance.

Key risks are (a) **portion estimation accuracy** (single-photo volume inference is unreliable), (b) **trust/privacy** (food photos can reveal sensitive context), and (c) **regulatory/claims creep**—we must stay clearly on the "wellness" side of FDA medical-device boundaries. Financially, a base-case model supports attractive unit economics: by **Year 3 ~32k paying subscribers** at **$17.99/mo** can drive **~$5–6M revenue**, with **~74% gross margin**, **blended CAC ~ $52**, and **LTV:CAC ~4.1x**. Estimated **break-even ~month 20–22** with **~$2.6M** upfront/early operating investment, contingent on holding **paid churn ≲6%** and maintaining recommendation quality to prevent post-novelty attrition.

</details>

## Quickstart

```python
import asyncio
from swarmcore import Swarm, researcher, writer

result = asyncio.run(Swarm(flow=researcher() >> writer()).run("AI agent trends in 2025"))
print(result.output)
```

`researcher` runs first (with web search). Its output is stored in shared context, which `writer` receives automatically.

Or build agents from scratch:

```python
from swarmcore import Agent, Swarm

r = Agent(name="researcher", instructions="Research the topic.", model="anthropic/claude-opus-4-6")
w = Agent(name="writer", instructions="Write a report from the research.", model="anthropic/claude-opus-4-6")

result = asyncio.run(Swarm(flow=r >> w).run("AI agent trends in 2025"))
```

## Agent factories

Pre-built factories for common roles. Zero-config defaults for prototyping, full override for production.

```python
from swarmcore import researcher, analyst, writer, editor, summarizer
```

| Factory | Default tools | Role |
|---|---|---|
| `researcher()` | `search_web` | Gathers information, finds data and sources |
| `analyst()` | — | Analyzes data, identifies insights and trends |
| `writer()` | — | Drafts structured content from context |
| `editor()` | — | Polishes and synthesizes multiple inputs |
| `summarizer()` | — | Condenses into an executive-level brief |

Every parameter is optional and keyword-only:

```python
# Zero-config
researcher()

# Fine-tuned
researcher(
    name="market_researcher",
    model="openai/gpt-4o",
    instructions="Focus on TAM and competitive landscape.",
    tools=[my_custom_search],   # replaces default search_web
    timeout=30.0,
)
```

Compose them into pipelines directly:

```python
flow = researcher() >> (analyst() | writer()) >> editor()
```

## Context management

Each agent in a flow receives context from prior steps automatically:

- **Previous step** — full output
- **Earlier steps** — summaries only
- **On demand** — agents call `expand_context` to retrieve any prior agent's full output

```
agent_1 >> agent_2 >> ... >> agent_10

agent_10 sees: agents 1-8 [SUMMARIES] + agent_9 [FULL]
               (can expand any earlier agent on demand)
```

Agents produce summaries via `<summary>` tags in their output. If omitted, the full output is used as both summary and detail.

## Flows

`>>` for sequential, `|` for parallel:

```python
planner >> writer                           # sequential
(researcher | critic) >> writer             # parallel then sequential
planner >> (researcher | critic) >> writer  # mixed
```

Parallel groups can contain multi-step sub-flows. `>>` binds tighter than `|`, so sub-chains compose naturally:

```python
(researcher >> analyst) | (critic >> editor) >> writer
```

```
researcher ──► analyst ──┐
                         ├──► writer
critic ──► editor ───────┘
```

Each branch runs its steps sequentially; branches run concurrently via `asyncio.gather`. Nesting is recursive — sub-flows can contain their own parallel groups.

Functional API:

```python
from swarmcore import chain, parallel

chain(planner, parallel(researcher, critic), writer)
chain(parallel(chain(researcher, analyst), chain(critic, editor)), writer)
```

Parallel agents share the same context snapshot — they don't see each other's outputs.

## Tools

Tools are plain Python functions. Type hints and docstrings are converted to tool schemas automatically. Sync and async both work.

```python
def search_web(query: str) -> str:
    """Search the web for information."""
    return results

agent = Agent(name="researcher", instructions="...", tools=[search_web])
```

## Models

Any [LiteLLM](https://docs.litellm.ai/)-compatible model:

```python
Agent(name="a", instructions="...", model="anthropic/claude-opus-4-6")
Agent(name="b", instructions="...", model="openai/gpt-4o")
Agent(name="c", instructions="...", model="ollama/llama3")
Agent(name="d", instructions="...", model="groq/llama-3.1-8b-instant")
```

## API

### `Agent(name, instructions, model, tools=None, timeout=None, max_retries=None, max_turns=None)`

| Param | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Identifier used in context keys |
| `instructions` | `str` | required | System prompt |
| `model` | `str` | `"anthropic/claude-opus-4-6"` | LiteLLM model string |
| `tools` | `list[Callable]` | `None` | Tool functions |
| `timeout` | `float \| None` | `None` | Per-agent LLM call timeout in seconds |
| `max_retries` | `int \| None` | `None` | Per-agent LLM retry count |
| `max_turns` | `int \| None` | `None` | Max tool-calling loop iterations |

### `Swarm(flow, hooks=None, timeout=None, max_retries=None)`

| Param | Type | Description |
|---|---|---|
| `flow` | `Flow` | Execution plan from operators or `chain()`/`parallel()` |
| `hooks` | `Hooks \| None` | Event hooks (e.g. `console_hooks()`) |
| `timeout` | `float \| None` | Default timeout for all agents |
| `max_retries` | `int \| None` | Default retry count for all agents |

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
