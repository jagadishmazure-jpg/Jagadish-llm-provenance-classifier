import numpy as np

from provenance.abstain import UNKNOWN, AbstainPolicy, choose_thresholds

LABELS = ["a", "b", "c"]
P = AbstainPolicy(min_prob=0.6, min_margin=0.2, min_words=30)


def test_confident_passage_is_attributed():
    assert P.decide(np.array([0.8, 0.1, 0.1]), LABELS, 60) == ("a", "attributed")


def test_short_text_abstains():
    label, reason = P.decide(np.array([0.95, 0.03, 0.02]), LABELS, 12)
    assert label == UNKNOWN and "too short" in reason


def test_low_confidence_abstains():
    label, reason = P.decide(np.array([0.5, 0.3, 0.2]), LABELS, 60)
    assert label == UNKNOWN and "low confidence" in reason


def test_close_runner_up_abstains():
    label, reason = P.decide(np.array([0.61, 0.45, 0.0]), LABELS, 60)
    assert label == UNKNOWN and "too close" in reason


def test_thresholds_reach_target_precision():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 3, n)
    conf = rng.uniform(0.34, 1.0, n)
    right = rng.random(n) < conf
    pred = np.where(right, y, (y + 1) % 3)
    probs = np.full((n, 3), 0.0)
    probs[np.arange(n), pred] = conf
    probs[np.arange(n), (pred + 1) % 3] = 1 - conf
    pol = choose_thresholds(probs, y, target_precision=0.9)
    keep = probs.max(1) >= pol.min_prob
    assert (pred[keep] == y[keep]).mean() >= 0.9 and keep.sum() >= 5


def test_impossible_target_falls_back_to_strict():
    probs = np.array([[0.9, 0.1]] * 10)
    y = np.array([1] * 10)  # always wrong
    assert choose_thresholds(probs, y).min_prob == 0.99
