"""Temperature scaling: one parameter, fitted on the validation split only."""

from __future__ import annotations

import numpy as np

from .metrics import nll


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Pick the temperature that minimises validation negative log-likelihood (grid search, deterministic)."""
    grid = np.round(np.arange(0.25, 5.01, 0.05), 2) if grid is None else grid
    losses = [nll(softmax(logits, t), y_idx) for t in grid]
    return float(grid[int(np.argmin(losses))])
