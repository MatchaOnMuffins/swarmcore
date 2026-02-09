from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def make_mock_response(
    content: str | None = "Mock response",
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Create a mock litellm ModelResponse."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage

    return response


@pytest.fixture
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch litellm.acompletion with an AsyncMock returning a standard response."""
    mock = AsyncMock(return_value=make_mock_response())
    monkeypatch.setattr("litellm.acompletion", mock)
    return mock
