from __future__ import annotations

from swarmcore.hooks import Event, EventType, Hooks


async def test_on_dispatches_to_specific_handler():
    received: list[Event] = []

    def handler(event: Event) -> None:
        received.append(event)

    hooks = Hooks()
    hooks.on(EventType.AGENT_START, handler)

    await hooks.emit(Event(EventType.AGENT_START, {"agent": "test"}))
    await hooks.emit(Event(EventType.AGENT_END, {"agent": "test"}))

    assert len(received) == 1
    assert received[0].type is EventType.AGENT_START


async def test_on_all_receives_every_event():
    received: list[EventType] = []

    def handler(event: Event) -> None:
        received.append(event.type)

    hooks = Hooks()
    hooks.on_all(handler)

    await hooks.emit(Event(EventType.SWARM_START))
    await hooks.emit(Event(EventType.AGENT_START))
    await hooks.emit(Event(EventType.SWARM_END))

    assert received == [
        EventType.SWARM_START,
        EventType.AGENT_START,
        EventType.SWARM_END,
    ]


async def test_is_active_false_when_empty():
    hooks = Hooks()
    assert hooks.is_active is False


async def test_is_active_true_with_specific_handler():
    hooks = Hooks()
    hooks.on(EventType.AGENT_START, lambda e: None)
    assert hooks.is_active is True


async def test_is_active_true_with_global_handler():
    hooks = Hooks()
    hooks.on_all(lambda e: None)
    assert hooks.is_active is True


async def test_sync_handler():
    called = []

    def handler(event: Event) -> None:
        called.append(event.type.value)

    hooks = Hooks()
    hooks.on_all(handler)

    await hooks.emit(Event(EventType.LLM_CALL_START))

    assert called == ["llm_call_start"]


async def test_async_handler():
    called = []

    async def handler(event: Event) -> None:
        called.append(event.type.value)

    hooks = Hooks()
    hooks.on_all(handler)

    await hooks.emit(Event(EventType.TOOL_CALL_END))

    assert called == ["tool_call_end"]


async def test_mixed_sync_async_handlers():
    called: list[str] = []

    def sync_handler(event: Event) -> None:
        called.append("sync")

    async def async_handler(event: Event) -> None:
        called.append("async")

    hooks = Hooks()
    hooks.on_all(sync_handler)
    hooks.on_all(async_handler)

    await hooks.emit(Event(EventType.STEP_START))

    assert "sync" in called
    assert "async" in called


async def test_handler_exception_is_swallowed():
    called_after = []

    def bad_handler(event: Event) -> None:
        raise ValueError("boom")

    def good_handler(event: Event) -> None:
        called_after.append(True)

    hooks = Hooks()
    hooks.on_all(bad_handler)
    hooks.on_all(good_handler)

    # Should not raise
    await hooks.emit(Event(EventType.AGENT_ERROR, {"error": "test"}))

    assert called_after == [True]


async def test_multiple_handlers_same_event():
    results: list[int] = []

    hooks = Hooks()
    hooks.on(EventType.SWARM_START, lambda e: results.append(1))
    hooks.on(EventType.SWARM_START, lambda e: results.append(2))

    await hooks.emit(Event(EventType.SWARM_START))

    assert results == [1, 2]
