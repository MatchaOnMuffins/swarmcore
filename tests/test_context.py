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


# --- Query method tests ---


def test_keys_empty():
    ctx = SharedContext()
    assert ctx.keys() == []


def test_keys_insertion_order():
    ctx = SharedContext()
    ctx.set("b", "B output")
    ctx.set("a", "A output")
    ctx.set("c", "C output")
    assert ctx.keys() == ["b", "a", "c"]


def test_search_matches():
    ctx = SharedContext()
    ctx.set("researcher", "Found AI trends.\nGrowth is exponential.")
    ctx.set("critic", "AI trends are overhyped.\nNeed more data.")
    results = ctx.search("AI trends")
    assert "researcher" in results
    assert "critic" in results
    assert results["researcher"] == ["Found AI trends."]
    assert results["critic"] == ["AI trends are overhyped."]


def test_search_no_matches():
    ctx = SharedContext()
    ctx.set("researcher", "Some output")
    assert ctx.search("nonexistent") == {}


def test_search_regex():
    ctx = SharedContext()
    ctx.set("agent", "line 1\nfoo123bar\nline 3")
    results = ctx.search(r"foo\d+bar")
    assert results == {"agent": ["foo123bar"]}


def test_search_invalid_regex_fallback():
    ctx = SharedContext()
    ctx.set("agent", "value is [bracket]")
    results = ctx.search("[bracket]")
    assert results == {"agent": ["value is [bracket]"]}


def test_entries_empty():
    ctx = SharedContext()
    assert ctx.entries() == []


def test_entries_returns_tuples():
    ctx = SharedContext()
    ctx.set("a", "Full A output", summary="A summary")
    ctx.set("b", "Full B")
    entries = ctx.entries()
    assert len(entries) == 2
    name, summary, full, count = entries[0]
    assert name == "a"
    assert summary == "A summary"
    assert full == "Full A output"
    assert count == len("Full A output")
    name2, summary2, full2, count2 = entries[1]
    assert name2 == "b"
    assert summary2 == "Full B"
    assert full2 == "Full B"
    assert count2 == len("Full B")
