"""Evaluation metrics, written out so every number in the README can be traced to code."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Better than the normal interval at small n or near 0/1."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred, strict=True):
        m[idx[t]][idx[p]] += 1
    return m


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        raise ValueError("empty")
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    scores = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores))


def ece(probs: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error on the top-class probability (equal-width bins)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def brier(probs: np.ndarray, y_idx: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs: np.ndarray, y_idx: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-12, 1.0))))
