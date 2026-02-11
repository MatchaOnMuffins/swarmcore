# SwarmCore

[![PyPI version](https://img.shields.io/pypi/v/swarmcore)](https://pypi.org/project/swarmcore/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/swarmcore)](https://pypi.org/project/swarmcore/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Multi-agent orchestration for Python. Compose AI agents into sequential and parallel flows with automatic context sharing.

<p align="center">
  <img src="assets/core_flow.gif" alt="SwarmCore flow demo" />
</p>

## Installation

```bash
pip install swarmcore
```

## Quickstart

```python
import asyncio
from swarmcore import Swarm, researcher, writer

result = asyncio.run(Swarm(flow=researcher() >> writer()).run("AI agent trends in 2025"))
print(result.output)
```

The default model is `anthropic/claude-opus-4-6` (requires `ANTHROPIC_API_KEY`). To use Google Gemini's free tier instead:

```python
flow = researcher(model="gemini/gemini-2.5-flash") >> writer(model="gemini/gemini-2.5-flash")
result = asyncio.run(Swarm(flow=flow).run("AI agent trends in 2025"))
```

## Features

- **Agent factories** — pre-built `researcher`, `analyst`, `writer`, `editor`, and `summarizer` with sensible defaults
- **Flow operators** — `>>` for sequential, `|` for parallel, compose freely
- **Automatic context** — each agent receives prior outputs automatically, with smart summarization
- **Any model** — works with any [LiteLLM](https://docs.litellm.ai/)-compatible provider (Anthropic, OpenAI, Gemini, Groq, Ollama, etc.)
- **Tool calling** — pass plain Python functions as tools, sync or async
- **Observability** — built-in `console_hooks()` for live progress, or wire up custom event hooks

## Agent factories

Pre-built factories for common workflow roles. Zero-config defaults for prototyping, full override for production.

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

Or build agents from scratch:

```python
from swarmcore import Agent

agent = Agent(name="researcher", instructions="Research the topic.", model="anthropic/claude-opus-4-6")
```

## Flows

`>>` for sequential, `|` for parallel:

```python
researcher() >> writer()                            # sequential
(analyst() | writer()) >> editor()                  # parallel then sequential
researcher() >> (analyst() | writer()) >> editor()  # mixed
```

Parallel groups can contain multi-step sub-flows:

```python
(researcher >> analyst) | (critic >> editor) >> writer
```

```
researcher ──► analyst ──┐
                         ├──► writer
critic ──► editor ───────┘
```

Each branch runs its steps sequentially; branches run concurrently via `asyncio.gather`.

Functional API:

```python
from swarmcore import chain, parallel

chain(planner, parallel(researcher, critic), writer)
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

Agents produce summaries via `<summary>` tags in their output. If omitted, the full output is used as both.

## Tools

Tools are plain Python functions. Type hints and docstrings are converted to tool schemas automatically.

```python
def search_web(query: str) -> str:
    """Search the web for information."""
    return results

agent = Agent(name="researcher", instructions="...", tools=[search_web])
```

Sync and async functions both work.

## Models

Any [LiteLLM](https://docs.litellm.ai/)-compatible model. Set the API key for your provider:

```bash
export ANTHROPIC_API_KEY=sk-...    # for anthropic/ models (default)
export GEMINI_API_KEY=...          # for gemini/ models (free tier available)
export OPENAI_API_KEY=sk-...       # for openai/ models
```

```python
researcher(model="anthropic/claude-opus-4-6")    # default
researcher(model="gemini/gemini-2.5-flash")      # free tier, no credit card needed
researcher(model="openai/gpt-4o")
researcher(model="groq/llama-3.1-8b-instant")
researcher(model="ollama/llama3")                # local, no API key needed
```

## Example: single agent vs. multi-agent flow

Both outputs below use the same model and prompt — the difference is orchestration.

> *Evaluate the opportunity for launching an AI-powered personal nutrition coach app that uses computer vision to analyze meals and provides real-time dietary recommendations.*

<details>
<summary><b>Single agent</b> — high-level strategic assessment</summary>

The opportunity for an AI-powered personal nutrition coach targeting health-conscious US millennials is compelling, driven by high smartphone usage, strong interest in personalized wellness, and frustration with manual calorie tracking. Computer vision–based meal analysis directly addresses a major pain point by reducing friction, while real-time dietary feedback aligns well with millennials' preference for on-demand, data-driven guidance integrated into daily life.

That said, the market is competitive and execution-sensitive: differentiation will require demonstrably accurate food recognition, culturally diverse meal coverage, and actionable recommendations that go beyond basic calorie counts. Trust will be critical, making privacy safeguards, transparent AI limitations, and careful positioning to avoid medical claims essential, alongside a clear monetization strategy (e.g., premium personalization or partnerships with fitness and health platforms).

</details>

<details>
<summary><b>Five-agent flow</b> — market sizing, unit economics, go-to-market strategy</summary>

The opportunity is meaningful if we position this as *frictionless meal logging plus actionable micro-coaching*, not "perfect" automated nutrition. The US mHealth apps market is estimated at **~$12.75B (2024)**, and meal occasions are increasingly "trackable" (Circana reports **86% of eating occasions are sourced from home**), creating high usage frequency for a camera-first workflow. Strategically, we should **launch with a hybrid CV + user-confirmation experience** (detect items, then prompt 1–2 quick portion inputs) and focus differentiation on (1) speed/low-friction capture, (2) trustworthy, goal-based recommendations, and (3) privacy-first handling of images. MVP is technically feasible in **~4–6 months** using pretrained vision + a constrained food ontology + human-in-the-loop QA; true defensibility likely requires **9–18 months** of data collection, model tuning, and nutrition governance.

Key risks are (a) **portion estimation accuracy** (single-photo volume inference is unreliable), (b) **trust/privacy** (food photos can reveal sensitive context), and (c) **regulatory/claims creep**—we must stay clearly on the "wellness" side of FDA medical-device boundaries. Financially, a base-case model supports attractive unit economics: by **Year 3 ~32k paying subscribers** at **$17.99/mo** can drive **~$5–6M revenue**, with **~74% gross margin**, **blended CAC ~ $52**, and **LTV:CAC ~4.1x**. Estimated **break-even ~month 20–22** with **~$2.6M** upfront/early operating investment, contingent on holding **paid churn ≲6%** and maintaining recommendation quality to prevent post-novelty attrition.

</details>

## API reference

### `Agent(name, instructions, model, tools, timeout, max_retries, max_turns)`

| Param | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Identifier used in context keys |
| `instructions` | `str` | required | System prompt |
| `model` | `str` | `"anthropic/claude-opus-4-6"` | LiteLLM model string |
| `tools` | `list[Callable]` | `None` | Tool functions |
| `timeout` | `float \| None` | `None` | Per-agent LLM call timeout in seconds |
| `max_retries` | `int \| None` | `None` | Per-agent LLM retry count |
| `max_turns` | `int \| None` | `None` | Max tool-calling loop iterations |

### `Swarm(flow, hooks, timeout, max_retries)`

| Param | Type | Default | Description |
|---|---|---|---|
| `flow` | `Flow` | required | Execution plan from operators or `chain()`/`parallel()` |
| `hooks` | `Hooks \| None` | `None` | Event hooks (e.g. `console_hooks()`) |
| `timeout` | `float \| None` | `None` | Default timeout for all agents |
| `max_retries` | `int \| None` | `None` | Default retry count for all agents |

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
