# CLAUDE.md

## Project Overview

SwarmCore is a Python framework for coordinating AI agents in workflows. It lets you define agents with specific instructions and models, compose them into sequential or parallel flows, and share context between them automatically. Any LiteLLM-compatible model works (OpenAI, Anthropic, Groq, Ollama, etc.).

**Version**: 0.1.1 | **License**: MIT | **Python**: >=3.11

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
pytest

# Run a specific test file
pytest tests/test_agent.py

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
pyright

# Build package
uv build
```

## Architecture

```
src/swarmcore/
├── __init__.py        # Public API exports
├── agent.py           # Agent class, tool schema conversion, LLM execution loop, >> and | operators
├── context.py         # SharedContext dual-storage (full + summaries) with query methods
├── context_tools.py   # Pull-mode context tool factories (list/get/search)
├── exceptions.py      # SwarmError, AgentError
├── flow.py            # Flow, chain(), parallel() — composable execution plans
├── models.py          # Pydantic data models (TokenUsage, AgentResult, SwarmResult)
└── swarm.py           # Swarm orchestrator, context_mode branching (push/pull), expand tool
```

### Core abstractions

- **Agent**: Wraps a single LLM call with instructions, model, and optional tools. Runs a tool-calling loop until the model returns a final text response. Stateless between runs.
- **Flow**: Immutable execution plan holding `list[Agent | list[Agent]]`. Built via `chain()`/`parallel()` functions or `>>` (sequential) / `|` (parallel) operators on agents.
- **Swarm**: Orchestrates agents via a `Flow` object. Runs steps sequentially or in parallel with `asyncio.gather()`, and maintains a `SharedContext`.
- **SharedContext**: Dual-storage (`_full` + `_summaries` dicts) for inter-agent communication. `set(key, value, summary=...)` stores both versions. `format_for_prompt(expand=...)` renders markdown sections (push mode). Query methods `keys()`, `search(pattern)`, `entries()` support pull-mode tooling.

### Flow syntax

Operator style (on Agent/Flow objects):
- `a >> b >> c` — sequential
- `a | b` — parallel
- `a >> (b | c) >> d` — mixed

Functional style:
- `chain(a, b, c)` — sequential
- `chain(parallel(a, b))` — parallel
- `chain(a, parallel(b, c), d)` — mixed

### Execution flow

`Swarm(flow, context_mode=)` receives a `Flow` object and a context mode (`"pull"` default, `"push"` for legacy). `swarm.run(task)` creates empty `SharedContext` → iterates `flow.steps`: if step is a single `Agent`, awaits `agent.run(task, context)`; if step is `list[Agent]`, uses `asyncio.gather()` → each agent receives context via push or pull mechanism → stores output in context → returns `SwarmResult(output, context, history)`.

### Context modes

**`context_mode="pull"` (default)** — agents pull context on demand via tools:

1. System prompt includes a **lightweight hint**: agent names + one-line summaries (no full outputs).
2. Three tools injected via `make_context_tools()` from `context_tools.py`:
   - `list_context()` — lists available agents, summaries, and char counts
   - `get_context(agent_name)` — retrieves full output from a prior agent
   - `search_context(query)` — regex search across all prior outputs
3. First step gets no tools or hint (no prior context exists).
4. `<summary>` tag parsing and graceful degradation still apply.

**`context_mode="push"`** — legacy tiered push system:

1. **Immediately preceding step** → full output shown (via `expand` set = `prev_step_names`)
2. **All earlier steps** → summaries shown (section headers get `(summary)` label)
3. **`expand_context` tool** → injected automatically when summarized entries exist. Agents can call `expand_context(agent_name="...")` at runtime to retrieve any prior agent's full output. Created by `_make_expand_tool()` closure over the `SharedContext`.
4. **Graceful degradation** → if no `<summary>` tags in response, full output is used as both summary and detail.

`agent.run()` accepts `extra_tools` and `context_hint` params — the Swarm uses these to inject tools/hints without modifying the Agent's permanent tool registry. Tools are merged into run-local `run_tools`/`run_schemas` dicts that don't persist between calls.

### Tool system

Tools are plain Python functions. `agent.py:_function_to_tool_schema()` introspects type hints and docstrings to generate OpenAI-compatible JSON tool schemas. Both sync and async functions are supported. Type mapping: `str→string`, `int→integer`, `float→number`, `bool→boolean`, `list→array`.

## Testing

- **Framework**: pytest + pytest-asyncio (auto mode)
- **Mocking**: `litellm.acompletion` is patched via a `mock_llm` fixture in `tests/conftest.py`
- **Pattern**: `make_mock_response()` factory creates mock LLM responses with configurable content and tool calls
- Tests cover: agent execution, context injection, tool calling (sync/async), flow construction (chain/parallel/operators), parallel/sequential swarm execution, push-mode tiered context (summary parsing, expand tool injection, graceful degradation), pull-mode context tools (list/get/search, hint injection, tool calling), error paths

## Code Conventions

- Full type annotations everywhere, including return types. Uses `X | None` syntax (not `Optional`).
- Async-first: core `run()` methods are async, no sync wrappers.
- Pydantic `BaseModel` for all data transfer objects.
- One responsibility per module. No circular imports.
- Custom exception hierarchy: `SwarmError` base, `AgentError` with `agent_name` field.
- `time.monotonic()` for duration measurement, token accumulation across tool calls.

## Dependencies

**Core**: `litellm` (LLM provider abstraction), `pydantic` (data validation)
**Dev**: `pytest`, `pytest-asyncio`, `ruff`, `pyright`
**Build**: `uv` (uv_build backend)

## CI/CD

GitHub Actions workflow (`.github/workflows/python-publish.yml`) publishes to PyPI on GitHub Release events. Uses `uv build` and PyPA's OIDC-based publish action.
