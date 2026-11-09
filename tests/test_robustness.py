import pytest

from provenance.robustness import drop_greeting_and_signoff, strip_format, truncate


def test_truncate_keeps_first_words():
    assert truncate("a b c d e", 3) == "a b c"


def test_truncate_rejects_zero():
    with pytest.raises(ValueError):
        truncate("a b", 0)


def test_strip_format_removes_markdown_and_placeholders():
    out = strip_format("**Tip:** call\n- one\n2. two\nDear [Name],")
    assert "**" not in out and "\n" not in out and "[" not in out and out.startswith("Tip:")


def test_drop_greeting_and_signoff():
    t = "Dear customer,\nYour claim is moving.\nKind regards,"
    assert drop_greeting_and_signoff(t) == "Your claim is moving."


def test_drop_leaves_plain_text_alone():
    assert drop_greeting_and_signoff("Line one.\nLine two.") == "Line one.\nLine two."


def test_strip_format_is_idempotent():
    t = "# Title\n**bold** text\n- item"
    assert strip_format(strip_format(t)) == strip_format(t)
