from __future__ import annotations

import io

from swarmcore.console import ConsoleReporter, console_hooks
from swarmcore.hooks import Event, EventType, Hooks


def _make_reporter(
    *, color: bool = True, verbose: bool = False
) -> tuple[ConsoleReporter, io.StringIO]:
    buf = io.StringIO()
    reporter = ConsoleReporter(color=color, verbose=verbose, file=buf)
    return reporter, buf


# -- All event types handled without error ----------------------------


def test_reporter_handles_all_events():
    """Every EventType can be dispatched without raising."""
    reporter, buf = _make_reporter()

    events = [
        Event(EventType.SWARM_START, {"task": "test", "step_count": 2}),
        Event(
            EventType.STEP_START,
            {"step_index": 0, "agents": ["a", "b"], "parallel": True},
        ),
        Event(EventType.AGENT_START, {"agent": "a", "task": "test"}),
        Event(EventType.LLM_CALL_START, {"agent": "a", "call_index": 0}),
        Event(
            EventType.LLM_CALL_END,
            {
                "agent": "a",
                "call_index": 0,
                "finish_reason": "tool_calls",
                "duration_seconds": 1.5,
                "total_tokens": 100,
            },
        ),
        Event(
            EventType.TOOL_CALL_START,
            {"agent": "a", "tool": "search", "arguments": {"q": "test"}},
        ),
        Event(
            EventType.TOOL_CALL_END,
            {"agent": "a", "tool": "search", "duration_seconds": 0.5},
        ),
        Event(EventType.LLM_CALL_START, {"agent": "a", "call_index": 1}),
        Event(
            EventType.LLM_CALL_END,
            {
                "agent": "a",
                "call_index": 1,
                "finish_reason": "stop",
                "duration_seconds": 2.0,
                "total_tokens": 200,
            },
        ),
        Event(EventType.AGENT_END, {"agent": "a", "duration_seconds": 4.0}),
        Event(EventType.STEP_END, {"step_index": 0}),
        Event(EventType.AGENT_ERROR, {"agent": "b", "error": "boom"}),
        Event(EventType.SWARM_END, {"duration_seconds": 5.0, "agent_count": 2}),
    ]

    for event in events:
        reporter(event)

    output = buf.getvalue()
    assert len(output) > 0


# -- Color disable ----------------------------------------------------


def test_color_disable():
    """With color=False, output contains no ANSI escape sequences."""
    reporter, buf = _make_reporter(color=False)

    reporter(
        Event(
            EventType.STEP_START,
            {"step_index": 0, "agents": ["researcher"], "parallel": False},
        )
    )
    reporter(Event(EventType.AGENT_START, {"agent": "researcher"}))
    reporter(
        Event(EventType.AGENT_END, {"agent": "researcher", "duration_seconds": 1.0})
    )
    reporter(Event(EventType.AGENT_ERROR, {"agent": "researcher", "error": "fail"}))

    output = buf.getvalue()
    assert "\033[" not in output


def test_color_enabled():
    """With color=True (default), output contains ANSI escape sequences."""
    reporter, buf = _make_reporter(color=True)

    reporter(Event(EventType.AGENT_END, {"agent": "a", "duration_seconds": 1.0}))

    output = buf.getvalue()
    assert "\033[" in output


# -- Verbose mode -----------------------------------------------------


def test_verbose_shows_args():
    """With verbose=True, tool call arguments appear in output."""
    reporter, buf = _make_reporter(verbose=True)

    reporter(
        Event(
            EventType.TOOL_CALL_START,
            {"agent": "a", "tool": "search", "arguments": {"query": "fusion"}},
        )
    )

    output = buf.getvalue()
    assert "query" in output
    assert "fusion" in output


def test_non_verbose_hides_args():
    """With verbose=False (default), tool call arguments are hidden."""
    reporter, buf = _make_reporter(verbose=False)

    reporter(
        Event(
            EventType.TOOL_CALL_START,
            {"agent": "a", "tool": "search", "arguments": {"query": "fusion"}},
        )
    )

    output = buf.getvalue()
    assert "search" in output
    assert "fusion" not in output


# -- Content checks ---------------------------------------------------


def test_step_start_parallel_label():
    reporter, buf = _make_reporter(color=False)

    reporter(
        Event(
            EventType.STEP_START,
            {"step_index": 0, "agents": ["a", "b", "c"], "parallel": True},
        )
    )

    output = buf.getvalue()
    assert "a | b | c" in output
    assert "(parallel)" in output


def test_step_start_sequential_label():
    reporter, buf = _make_reporter(color=False)

    reporter(
        Event(
            EventType.STEP_START,
            {"step_index": 1, "agents": ["writer"], "parallel": False},
        )
    )

    output = buf.getvalue()
    assert "Step 2: writer" in output
    assert "(parallel)" not in output


def test_llm_call_end_tool_calls_indicator():
    reporter, buf = _make_reporter(color=False)

    reporter(
        Event(
            EventType.LLM_CALL_END,
            {
                "agent": "a",
                "call_index": 0,
                "finish_reason": "tool_calls",
                "duration_seconds": 1.0,
                "total_tokens": 50,
            },
        )
    )

    output = buf.getvalue()
    assert "tool_calls" in output


def test_llm_call_end_stop_no_indicator():
    reporter, buf = _make_reporter(color=False)

    reporter(
        Event(
            EventType.LLM_CALL_END,
            {
                "agent": "a",
                "call_index": 0,
                "finish_reason": "stop",
                "duration_seconds": 1.0,
                "total_tokens": 50,
            },
        )
    )

    output = buf.getvalue()
    assert "tool_calls" not in output


def test_swarm_end_summary():
    reporter, buf = _make_reporter(color=False)

    reporter(Event(EventType.SWARM_END, {"duration_seconds": 10.5, "agent_count": 4}))

    output = buf.getvalue()
    assert "4 agents" in output
    assert "10.5s" in output


# -- Factory function -------------------------------------------------


def test_console_hooks_factory():
    hooks = console_hooks()
    assert isinstance(hooks, Hooks)
    assert hooks.is_active is True


def test_console_hooks_passes_options():
    buf = io.StringIO()
    hooks = console_hooks(color=False, verbose=True, file=buf)
    assert hooks.is_active is True
