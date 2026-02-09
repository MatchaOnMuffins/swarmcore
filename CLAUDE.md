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
├── agent.py         # Agent class, tool schema conversion, LLM execution loop
├── context.py       # SharedContext key-value store for inter-agent communication
├── exceptions.py    # SwarmError, AgentError
├── models.py        # Pydantic data models (TokenUsage, AgentResult, SwarmResult)
└── swarm.py         # Swarm orchestrator, flow parser
```

### Core abstractions

- **Agent**: Wraps a single LLM call with instructions, model, and optional tools. Runs a tool-calling loop until the model returns a final text response. Stateless between runs.
- **Swarm**: Orchestrates multiple agents via a flow string. Parses the flow, validates agent names, runs steps sequentially or in parallel with `asyncio.gather()`, and maintains a `SharedContext`.
- **SharedContext**: Simple `dict[str, str]` store. Each agent's output is stored under its name. Context is formatted as markdown sections and injected into subsequent agents' system prompts.

### Flow syntax

- `a >> b >> c` — sequential
- `[a, b]` — parallel
- `a >> [b, c] >> d` — mixed

### Execution flow

`swarm.run(task)` → creates empty `SharedContext` → for each step: if sequential, awaits `agent.run(task, context)`; if parallel, uses `asyncio.gather()` → each agent injects context into system prompt, calls `litellm.acompletion()`, loops on tool calls → stores output in context → returns `SwarmResult(output, context, history)`.

### Tool system

Tools are plain Python functions. `agent.py:_function_to_tool_schema()` introspects type hints and docstrings to generate OpenAI-compatible JSON tool schemas. Both sync and async functions are supported. Type mapping: `str→string`, `int→integer`, `float→number`, `bool→boolean`, `list→array`.

## Testing

- **Framework**: pytest + pytest-asyncio (auto mode)
- **Mocking**: `litellm.acompletion` is patched via a `mock_llm` fixture in `tests/conftest.py`
- **Pattern**: `make_mock_response()` factory creates mock LLM responses with configurable content and tool calls
- Tests cover: agent execution, context injection, tool calling (sync/async), flow parsing, parallel/sequential swarm execution, error paths

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
