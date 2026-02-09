from swarmcore.context import SharedContext


def test_set_and_get():
    ctx = SharedContext()
    ctx.set("researcher", "Some notes")
    assert ctx.get("researcher") == "Some notes"


def test_get_missing_key():
    ctx = SharedContext()
    assert ctx.get("nonexistent") is None


def test_format_for_prompt_empty():
    ctx = SharedContext()
    assert ctx.format_for_prompt() == ""


def test_format_for_prompt_single():
    ctx = SharedContext()
    ctx.set("researcher", "AI trends are growing.")
    result = ctx.format_for_prompt()
    assert "## researcher" in result
    assert "AI trends are growing." in result


def test_format_for_prompt_multiple():
    ctx = SharedContext()
    ctx.set("researcher", "Research notes")
    ctx.set("critic", "Critical feedback")
    result = ctx.format_for_prompt()
    assert "## researcher" in result
    assert "## critic" in result
    assert "Research notes" in result
    assert "Critical feedback" in result
    # Sections separated by double newline
    assert "\n\n" in result


def test_to_dict():
    ctx = SharedContext()
    ctx.set("a", "value_a")
    ctx.set("b", "value_b")
    d = ctx.to_dict()
    assert d == {"a": "value_a", "b": "value_b"}
    # Modifying the returned dict should not affect the context
    d["c"] = "value_c"
    assert ctx.get("c") is None


def test_overwrite():
    ctx = SharedContext()
    ctx.set("key", "first")
    ctx.set("key", "second")
    assert ctx.get("key") == "second"
