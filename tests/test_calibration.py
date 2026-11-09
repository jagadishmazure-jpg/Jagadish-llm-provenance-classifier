import numpy as np
import pytest

from provenance.calibration import fit_temperature, softmax


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        softmax(np.zeros((1, 2)), 0)


def test_higher_temperature_flattens():
    lg = np.array([[4.0, 0.0]])
    assert softmax(lg, 3.0).max() < softmax(lg, 1.0).max()


def test_overconfident_logits_get_temperature_above_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    correct = rng.random(400) < 0.7  # 70% right, but logits say ~100% sure
    pred = np.where(correct, y, 1 - y)
    lg = np.zeros((400, 2))
    lg[np.arange(400), pred] = 10.0
    assert fit_temperature(lg, y) > 2.0


def test_fit_is_deterministic():
    lg = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 0.5]])
    y = np.array([0, 1, 1])
    assert fit_temperature(lg, y) == fit_temperature(lg, y)
