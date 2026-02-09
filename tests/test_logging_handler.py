from __future__ import annotations

import logging

import pytest

from swarmcore.hooks import Event, EventType, Hooks
from swarmcore.logging import LoggingHandler, enable_logging


def test_logging_handler_info_events(caplog: pytest.LogCaptureFixture):
    handler = LoggingHandler()

    with caplog.at_level(logging.DEBUG, logger="swarmcore"):
        handler(Event(EventType.SWARM_START, {"task": "test"}))
        handler(Event(EventType.AGENT_START, {"agent": "a"}))
        handler(Event(EventType.SWARM_END, {"duration_seconds": 1.0}))

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_records) == 3


def test_logging_handler_debug_events(caplog: pytest.LogCaptureFixture):
    handler = LoggingHandler()

    with caplog.at_level(logging.DEBUG, logger="swarmcore"):
        handler(Event(EventType.LLM_CALL_START, {"agent": "a", "call_index": 0}))
        handler(Event(EventType.TOOL_CALL_END, {"agent": "a", "tool": "search"}))

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debug_records) == 2


def test_logging_handler_error_events(caplog: pytest.LogCaptureFixture):
    handler = LoggingHandler()

    with caplog.at_level(logging.DEBUG, logger="swarmcore"):
        handler(Event(EventType.AGENT_ERROR, {"agent": "a", "error": "boom"}))

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1


def test_enable_logging_returns_hooks():
    hooks = enable_logging(level=logging.DEBUG)

    assert isinstance(hooks, Hooks)
    assert hooks.is_active is True
