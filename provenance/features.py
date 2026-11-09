"""Hand-written stylometry features. Cheap, explainable, and model-agnostic."""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_WORD = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"[^.!?\n]+[.!?]?")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")

FEATURE_NAMES = [
    "n_words_log", "mean_sentence_words", "sd_sentence_words", "mean_word_len", "type_token_ratio",
    "comma_rate", "exclaim_rate", "question_rate", "colon_rate", "semicolon_rate", "dash_rate",
    "newline_rate", "blank_line_rate", "bullet_lines", "numbered_lines", "markdown_bold", "heading_lines",
    "placeholder_brackets", "quote_rate", "upper_ratio", "digit_ratio", "emoji_rate", "hashtag_rate",
    "contraction_rate", "first_person_rate", "second_person_rate", "starts_dear", "starts_subject",
    "starts_qa", "starts_preamble", "ends_signoff", "truncated_end",
]


def stylometry(text: str) -> list[float]:
    words = _WORD.findall(text)
    n = max(len(words), 1)
    chars = max(len(text), 1)
    sents = [s for s in _SENT.findall(text) if _WORD.search(s)]
    slen = [len(_WORD.findall(s)) for s in sents] or [0]
    lines = text.splitlines() or [""]
    low = [w.lower() for w in words]
    first = text.lstrip()[:40].lower()
    last = text.rstrip()[-60:].lower()
    return [
        float(np.log1p(len(words))),
        float(np.mean(slen)),
        float(np.std(slen)),
        float(np.mean([len(w) for w in words])) if words else 0.0,
        len(set(low)) / n,
        text.count(",") / n,
        text.count("!") / n,
        text.count("?") / n,
        text.count(":") / n,
        text.count(";") / n,
        (text.count("-") + text.count("\u2014")) / n,
        text.count("\n") / n,
        text.count("\n\n") / n,
        float(sum(1 for ln in lines if re.match(r"\s*[-*\u2022]\s", ln))),
        float(sum(1 for ln in lines if re.match(r"\s*\d+[.)]\s", ln))),
        float(text.count("**") // 2),
        float(sum(1 for ln in lines if re.match(r"\s*#+\s", ln))),
        float(len(re.findall(r"\[[^\]]{1,40}\]", text))),
        (text.count('"') + text.count("\u201c")) / n,
        sum(c.isupper() for c in text) / chars,
        sum(c.isdigit() for c in text) / chars,
        len(_EMOJI.findall(text)) / n,
        text.count("#") / n if re.search(r"#\w", text) else 0.0,
        sum(1 for w in words if "'" in w) / n,
        sum(1 for w in low if w in {"i", "we", "our", "us", "my", "me"}) / n,
        sum(1 for w in low if w in {"you", "your", "yours"}) / n,
        float(first.startswith("dear")),
        float(first.startswith("subject")),
        float(first.startswith(("q:", "**q", "faq", "question"))),
        float(first.startswith(("certainly", "sure", "here", "of course", "absolutely"))),
        float(bool(re.search(r"(regards|sincerely|best wishes|warm wishes|\[your name\])", last))),
        float(not text.rstrip().endswith((".", "!", "?", ")", "]", '"', "*"))),
    ]


class Stylometry(BaseEstimator, TransformerMixin):
    """scikit-learn transformer wrapper so stylometry can sit in a FeatureUnion."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn naming
        return self

    def transform(self, X):  # noqa: N803
        return np.array([stylometry(t) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
