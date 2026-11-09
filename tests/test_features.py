import math

from provenance.features import FEATURE_NAMES, Stylometry, stylometry


def f(text, name):
    return stylometry(text)[FEATURE_NAMES.index(name)]


def test_vector_matches_names():
    assert len(stylometry("Hello there. How are you?")) == len(FEATURE_NAMES)


def test_empty_text_is_safe():
    assert not any(math.isnan(v) for v in stylometry(""))


def test_opening_and_closing_flags():
    cases = [("Dear customer,\nThanks.", "starts_dear"), ("Subject: Your claim\nHello", "starts_subject"),
             ("Certainly! Here is the answer.", "starts_preamble"),
             ("Q: What is an excess?\nA: The part you pay.", "starts_qa"),
             ("Thanks for waiting.\nBest regards,\n[Your Name]", "ends_signoff")]
    for text, name in cases:
        assert f(text, name) == 1.0, name
        assert f("Plain text here.", name) == 0.0, name


def test_list_and_markdown_counts():
    t = "**Tips**\n- one\n- two\n1. first\n2) second\n# Heading"
    assert f(t, "bullet_lines") == 2 and f(t, "numbered_lines") == 2
    assert f(t, "markdown_bold") == 1 and f(t, "heading_lines") == 1


def test_placeholders_and_truncation():
    assert f("Dear [Customer Name], see [Policy Number]", "placeholder_brackets") == 2
    assert f("This sentence was cut off in the mid", "truncated_end") == 1.0
    assert f("This one ends.", "truncated_end") == 0.0


def test_transformer_shape():
    X = Stylometry().fit_transform(["a b c.", "d e f!"])
    assert X.shape == (2, len(FEATURE_NAMES))
