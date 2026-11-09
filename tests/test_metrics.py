import numpy as np
import pytest

from provenance.metrics import accuracy, brier, confusion, ece, macro_f1, nll, wilson


def test_wilson_zero_of_ten():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)


def test_wilson_all_of_ten_and_symmetry():
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-4)
    a, b = wilson(3, 10)
    c, d = wilson(7, 10)
    assert a == pytest.approx(1 - d) and b == pytest.approx(1 - c)


def test_wilson_input_checks():
    with pytest.raises(ValueError):
        wilson(1, 0)
    with pytest.raises(ValueError):
        wilson(11, 10)


def test_confusion_and_accuracy():
    m = confusion(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert m == [[1, 1], [0, 1]]
    assert accuracy(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 3)


def test_macro_f1_perfect_and_zero():
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    assert macro_f1(["a", "b"], ["b", "a"], ["a", "b"]) == 0.0


def test_ece_zero_when_confident_and_right():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(p, np.array([0, 1])) == 0.0


def test_ece_large_when_confident_and_wrong():
    p = np.array([[0.99, 0.01]] * 4)
    assert ece(p, np.array([1, 1, 1, 1])) == pytest.approx(0.99)


def test_brier_and_nll():
    p = np.array([[0.5, 0.5]])
    assert brier(p, np.array([0])) == pytest.approx(0.5)
    assert nll(p, np.array([0])) == pytest.approx(np.log(2))
