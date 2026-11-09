"""Abstain logic: say "unknown model" instead of guessing.

A passage is attributed only when the calibrated top probability is at least `min_prob`
AND the gap to the runner-up is at least `min_margin` AND the passage has at least
`min_words` words. Thresholds are chosen on the validation split (known models only);
the out-of-set model is never used to tune them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNKNOWN = "unknown"


@dataclass(frozen=True)
class AbstainPolicy:
    min_prob: float = 0.6
    min_margin: float = 0.2
    min_words: int = 30

    def decide(self, probs: np.ndarray, labels: list[str], n_words: int) -> tuple[str, str]:
        """Return (label or 'unknown', reason)."""
        order = np.argsort(probs)[::-1]
        top, second = float(probs[order[0]]), float(probs[order[1]]) if len(probs) > 1 else 0.0
        if n_words < self.min_words:
            return UNKNOWN, f"too short ({n_words} words < {self.min_words})"
        if top < self.min_prob:
            return UNKNOWN, f"low confidence ({top:.2f} < {self.min_prob:.2f})"
        if top - second < self.min_margin:
            return UNKNOWN, f"two models too close (margin {top - second:.2f} < {self.min_margin:.2f})"
        return labels[int(order[0])], "attributed"


def choose_thresholds(probs: np.ndarray, y_idx: np.ndarray, target_precision: float = 0.95,
                      min_words: int = 30) -> AbstainPolicy:
    """Smallest min_prob (max coverage) whose accepted validation passages reach target precision.

    Margin is fixed at a quarter of min_prob. If no threshold reaches the target, the strictest
    grid value is returned, which means the system mostly abstains (an honest outcome).
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    srt = np.sort(probs, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    best = None
    for t in np.round(np.arange(0.40, 0.991, 0.01), 2):
        m = round(float(t) / 4, 3)
        keep = (conf >= t) & (margin >= m)
        if keep.sum() >= 5 and (pred[keep] == y_idx[keep]).mean() >= target_precision:
            best = AbstainPolicy(float(t), m, min_words)
            break
    return best or AbstainPolicy(0.99, 0.25, min_words)
