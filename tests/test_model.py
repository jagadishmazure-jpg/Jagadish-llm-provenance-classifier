import numpy as np
import pytest

from provenance.corpus import validate
from provenance.model import ProvenanceModel, build_features


def P(i, label, text):
    return validate({"id": f"{label}:{i}", "label": label, "prompt_id": f"p{i}", "topic_id": i, "task": "t",
                     "split": "train", "text": text})


def test_needs_two_models():
    with pytest.raises(ValueError, match="two models"):
        ProvenanceModel().fit([P(0, "a", "one two three"), P(1, "a", "four five six")])


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        build_features("bert")


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        ProvenanceModel().predict_proba(["hello"])


def test_toy_style_is_learned():
    a = [P(i, "dear", f"Dear customer, thank you for item {i}. Kind regards, the team.") for i in range(12)]
    b = [P(i + 50, "bullets", f"- tip {i}\n- check\n- call us now!") for i in range(12)]
    m = ProvenanceModel(variant="stylometry").fit(a + b)
    assert m.attribute(["- tip x\n- check this\n- call!"])[0]["scores"]["bullets"] > 0.5


def test_real_probabilities_are_valid(fitted):
    m, data = fitted
    p = m.predict_proba([x.text for x in data[:20]])
    assert p.shape == (20, 4) and np.allclose(p.sum(1), 1.0)


def test_calibration_sets_positive_temperature_and_policy(fitted):
    m, _ = fitted
    assert m.temperature > 0
    assert 0.4 <= m.policy.min_prob <= 0.99


def test_attribute_output_shape(fitted):
    m, data = fitted
    a = m.attribute([data[0].text])[0]
    assert set(a) == {"label", "reason", "top_prob", "scores"} and set(a["scores"]) == set(m.labels)


def test_top_features_are_named(fitted):
    m, _ = fitted
    feats = m.top_features(m.labels[0], 5)
    assert len(feats) == 5 and all("__" in f for f in feats)
