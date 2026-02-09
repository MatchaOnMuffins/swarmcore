from swarmcore.swarm import _parse_structured_output


def test_valid_tags():
    output = "<summary>Short summary here.</summary>\nDetailed output follows."
    summary, detail = _parse_structured_output(output)
    assert summary == "Short summary here."
    assert detail == "Detailed output follows."


def test_no_tags_graceful_degradation():
    output = "Just a plain response with no tags."
    summary, detail = _parse_structured_output(output)
    assert summary == output
    assert detail == output


def test_empty_detail():
    output = "<summary>Only summary, no detail.</summary>"
    summary, detail = _parse_structured_output(output)
    assert summary == "Only summary, no detail."
    assert detail == ""


def test_whitespace_around_tags():
    output = "  <summary>  Spaced summary  </summary>  \nSome detail  "
    summary, detail = _parse_structured_output(output)
    assert summary == "Spaced summary"
    assert detail == "Some detail"


def test_multiline_summary():
    output = (
        "<summary>Line one of summary.\n"
        "Line two of summary.</summary>\n"
        "The detailed content."
    )
    summary, detail = _parse_structured_output(output)
    assert "Line one" in summary
    assert "Line two" in summary
    assert detail == "The detailed content."


def test_summary_in_middle():
    output = "Preamble.\n<summary>Mid summary.</summary>\nAftermath."
    summary, detail = _parse_structured_output(output)
    assert summary == "Mid summary."
    assert "Preamble." in detail
    assert "Aftermath." in detail
    assert "<summary>" not in detail
