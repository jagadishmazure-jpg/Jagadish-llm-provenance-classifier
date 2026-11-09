"""Cheap, deterministic perturbations. Model-made paraphrases and style-copy attacks live in the corpus."""

from __future__ import annotations

import re


def truncate(text: str, n_words: int) -> str:
    """Keep the first n words (what an auditor sees in a snippet or a cut-off export)."""
    if n_words <= 0:
        raise ValueError("n_words must be positive")
    return " ".join(text.split()[:n_words])


def strip_format(text: str) -> str:
    """Remove markdown, bullets, line breaks and placeholders: a copy-paste into a plain web form."""
    t = re.sub(r"\*\*|__|^#+\s*|`", "", text, flags=re.M)
    t = re.sub(r"^\s*([-*\u2022]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"\[[^\]]{1,40}\]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def drop_greeting_and_signoff(text: str) -> str:
    """Remove the first and last line when they look like a greeting or a sign-off (light human edit)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and re.match(r"\s*(dear|hi|hello|subject)\b", lines[0], re.I):
        lines = lines[1:]
    if lines and re.search(r"(regards|sincerely|wishes|\[your name\]|team)\W*$", lines[-1], re.I):
        lines = lines[:-1]
    return "\n".join(lines)
