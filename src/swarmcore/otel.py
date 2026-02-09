"""Optional OpenTelemetry integration for SwarmCore.

Requires the ``opentelemetry-api`` and ``opentelemetry-sdk`` packages::

    pip install swarmcore[otel]
"""

from __future__ import annotations

from typing import Any

try:
    from opentelemetry import trace  # pyright: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "OpenTelemetry packages are required for OTelHandler. "
        "Install them with: pip install swarmcore[otel]"
    ) from exc

from swarmcore.hooks import Event, EventType

_tracer = trace.get_tracer("swarmcore")


class OTelHandler:
    """Hook handler that creates OpenTelemetry spans for swarmcore events.

    Produces a span hierarchy::

        swarm.run
        └── swarm.step[0]
            ├── agent.researcher
            │   ├── llm.call[0]
            │   └── tool.get_weather
            └── agent.writer
                └── llm.call[0]

    Handles parallel agents correctly by keying active spans on composite
    string identifiers rather than assuming sequential execution.
    """

    def __init__(self) -> None:
        self._spans: dict[str, Any] = {}

    def _key(self, *parts: str | int) -> str:
        return ":".join(str(p) for p in parts)

    def __call__(self, event: Event) -> None:
        data = event.data

        if event.type is EventType.SWARM_START:
            span = _tracer.start_span("swarm.run")
            span.set_attribute("swarm.task", data.get("task", ""))
            span.set_attribute("swarm.step_count", data.get("step_count", 0))
            self._spans["swarm"] = span

        elif event.type is EventType.SWARM_END:
            span = self._spans.pop("swarm", None)
            if span:
                span.set_attribute(
                    "swarm.duration_seconds", data.get("duration_seconds", 0)
                )
                span.end()

        elif event.type is EventType.STEP_START:
            idx = data.get("step_index", 0)
            parent = self._spans.get("swarm")
            ctx = trace.set_span_in_context(parent) if parent else None
            span = _tracer.start_span(f"swarm.step[{idx}]", context=ctx)
            span.set_attribute("step.parallel", data.get("parallel", False))
            self._spans[self._key("step", idx)] = span

        elif event.type is EventType.STEP_END:
            idx = data.get("step_index", 0)
            span = self._spans.pop(self._key("step", idx), None)
            if span:
                span.end()

        elif event.type is EventType.AGENT_START:
            agent = data.get("agent", "")
            # Find the current step span as parent
            parent = None
            for key in reversed(list(self._spans)):
                if key.startswith("step:"):
                    parent = self._spans[key]
                    break
            ctx = trace.set_span_in_context(parent) if parent else None
            span = _tracer.start_span(f"agent.{agent}", context=ctx)
            span.set_attribute("agent.task", data.get("task", ""))
            self._spans[self._key("agent", agent)] = span

        elif event.type is EventType.AGENT_END:
            agent = data.get("agent", "")
            span = self._spans.pop(self._key("agent", agent), None)
            if span:
                span.set_attribute(
                    "agent.duration_seconds", data.get("duration_seconds", 0)
                )
                span.end()

        elif event.type is EventType.AGENT_ERROR:
            agent = data.get("agent", "")
            span = self._spans.pop(self._key("agent", agent), None)
            if span:
                span.set_attribute("error", True)
                span.set_attribute("error.message", data.get("error", ""))
                span.end()

        elif event.type is EventType.LLM_CALL_START:
            agent = data.get("agent", "")
            idx = data.get("call_index", 0)
            parent = self._spans.get(self._key("agent", agent))
            ctx = trace.set_span_in_context(parent) if parent else None
            span = _tracer.start_span(f"llm.call[{idx}]", context=ctx)
            self._spans[self._key("llm", agent, idx)] = span

        elif event.type is EventType.LLM_CALL_END:
            agent = data.get("agent", "")
            idx = data.get("call_index", 0)
            span = self._spans.pop(self._key("llm", agent, idx), None)
            if span:
                span.set_attribute("llm.finish_reason", data.get("finish_reason", ""))
                span.set_attribute(
                    "llm.duration_seconds", data.get("duration_seconds", 0)
                )
                span.end()

        elif event.type is EventType.TOOL_CALL_START:
            agent = data.get("agent", "")
            tool = data.get("tool", "")
            parent = self._spans.get(self._key("agent", agent))
            ctx = trace.set_span_in_context(parent) if parent else None
            span = _tracer.start_span(f"tool.{tool}", context=ctx)
            self._spans[self._key("tool", agent, tool)] = span

        elif event.type is EventType.TOOL_CALL_END:
            agent = data.get("agent", "")
            tool = data.get("tool", "")
            span = self._spans.pop(self._key("tool", agent, tool), None)
            if span:
                span.set_attribute(
                    "tool.duration_seconds", data.get("duration_seconds", 0)
                )
                span.end()
