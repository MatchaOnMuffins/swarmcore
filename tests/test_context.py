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


# --- Tiered context tests ---


def test_set_with_summary():
    ctx = SharedContext()
    ctx.set("researcher", "Full detailed output here.", summary="Short summary.")
    assert ctx.get("researcher") == "Full detailed output here."
    assert ctx.get_summary("researcher") == "Short summary."


def test_set_without_summary_uses_full():
    ctx = SharedContext()
    ctx.set("researcher", "Full output")
    assert ctx.get_summary("researcher") == "Full output"


def test_get_summary_missing_key():
    ctx = SharedContext()
    assert ctx.get_summary("nonexistent") is None


def test_format_for_prompt_expand_specific_keys():
    ctx = SharedContext()
    ctx.set("a", "Full A", summary="Summary A")
    ctx.set("b", "Full B", summary="Summary B")
    ctx.set("c", "Full C", summary="Summary C")

    result = ctx.format_for_prompt(expand={"b"})
    assert "Summary A" in result
    assert "Full A" not in result
    assert "## a (summary)" in result
    assert "Full B" in result
    assert "Summary B" not in result
    assert "## b\n" in result
    assert "Summary C" in result
    assert "Full C" not in result
    assert "## c (summary)" in result


def test_format_for_prompt_expand_none_shows_all_full():
    ctx = SharedContext()
    ctx.set("a", "Full A", summary="Summary A")
    ctx.set("b", "Full B", summary="Summary B")

    result = ctx.format_for_prompt(expand=None)
    assert "Full A" in result
    assert "Full B" in result
    assert "Summary A" not in result
    assert "Summary B" not in result


def test_format_for_prompt_expand_empty_set_shows_all_summaries():
    ctx = SharedContext()
    ctx.set("a", "Full A", summary="Summary A")
    ctx.set("b", "Full B", summary="Summary B")

    result = ctx.format_for_prompt(expand=set())
    assert "Summary A" in result
    assert "Summary B" in result
    assert "Full A" not in result
    assert "Full B" not in result
    assert "## a (summary)" in result
    assert "## b (summary)" in result


def test_format_for_prompt_expand_none_no_summary_labels():
    ctx = SharedContext()
    ctx.set("a", "Full A", summary="Summary A")

    result = ctx.format_for_prompt(expand=None)
    assert "(summary)" not in result
