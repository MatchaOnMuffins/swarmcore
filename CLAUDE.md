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
├── __init__.py      # Public API exports
├── agent.py         # Agent class, tool schema conversion, LLM execution loop, >> and | operators
├── context.py       # SharedContext dual-storage (full + summaries) for inter-agent communication
├── exceptions.py    # SwarmError, AgentError
├── flow.py          # Flow, chain(), parallel() — composable execution plans
├── models.py        # Pydantic data models (TokenUsage, AgentResult, SwarmResult)
└── swarm.py         # Swarm orchestrator, executes Flow steps, tiered context wiring, expand tool
```

### Core abstractions

- **Agent**: Wraps a single LLM call with instructions, model, and optional tools. Runs a tool-calling loop until the model returns a final text response. Stateless between runs.
- **Flow**: Immutable execution plan holding `list[Agent | list[Agent]]`. Built via `chain()`/`parallel()` functions or `>>` (sequential) / `|` (parallel) operators on agents.
- **Swarm**: Orchestrates agents via a `Flow` object. Runs steps sequentially or in parallel with `asyncio.gather()`, and maintains a `SharedContext`.
- **SharedContext**: Dual-storage (`_full` + `_summaries` dicts) for inter-agent communication. `set(key, value, summary=...)` stores both versions. `format_for_prompt(expand=...)` renders markdown sections — keys in `expand` show full output, the rest show summaries with `(summary)` label in the header. `expand=None` shows all full (backward compat).

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

`Swarm(flow)` receives a `Flow` object → `swarm.run(task)` creates empty `SharedContext` → iterates `flow.steps`: if step is a single `Agent`, awaits `agent.run(task, context)`; if step is `list[Agent]`, uses `asyncio.gather()` → each agent injects context into system prompt, calls `litellm.acompletion()`, loops on tool calls → stores output in context → returns `SwarmResult(output, context, history)`.

### Tiered context system

Agents produce structured output with `<summary>...</summary>` tags. The Swarm parses these via `_parse_structured_output()` (regex in `swarm.py`) and stores both versions in `SharedContext`. The tiering logic:

1. **Immediately preceding step** → full output shown (via `expand` set = `prev_step_names`)
2. **All earlier steps** → summaries shown (section headers get `(summary)` label)
3. **`expand_context` tool** → injected automatically when summarized entries exist. Agents can call `expand_context(agent_name="...")` at runtime to retrieve any prior agent's full output. Created by `_make_expand_tool()` closure over the `SharedContext`.
4. **Graceful degradation** → if no `<summary>` tags in response, full output is used as both summary and detail.

`agent.run()` accepts `extra_tools` param — the Swarm uses this to inject the expand tool without modifying the Agent's permanent tool registry. Tools are merged into run-local `run_tools`/`run_schemas` dicts that don't persist between calls.

### Tool system

Tools are plain Python functions. `agent.py:_function_to_tool_schema()` introspects type hints and docstrings to generate OpenAI-compatible JSON tool schemas. Both sync and async functions are supported. Type mapping: `str→string`, `int→integer`, `float→number`, `bool→boolean`, `list→array`.

## Testing

- **Framework**: pytest + pytest-asyncio (auto mode)
- **Mocking**: `litellm.acompletion` is patched via a `mock_llm` fixture in `tests/conftest.py`
- **Pattern**: `make_mock_response()` factory creates mock LLM responses with configurable content and tool calls
- Tests cover: agent execution, context injection, tool calling (sync/async), flow construction (chain/parallel/operators), parallel/sequential swarm execution, tiered context (summary parsing, expand tool injection, graceful degradation), error paths

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
