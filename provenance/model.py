"""Feature-based baseline: character + word TF-IDF and stylometry into a logistic regression."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from .abstain import AbstainPolicy, choose_thresholds
from .calibration import fit_temperature, softmax
from .corpus import Passage
from .features import Stylometry

VARIANTS = ("full", "tfidf", "stylometry")


def build_features(variant: str = "full") -> FeatureUnion:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    parts = []
    if variant in ("full", "tfidf"):
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                              sublinear_tf=True, max_features=60000)))
        parts.append(("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                              sublinear_tf=True, lowercase=True, max_features=30000)))
    if variant in ("full", "stylometry"):
        parts.append(("style", Pipeline([("f", Stylometry()), ("s", StandardScaler())])))
    return FeatureUnion(parts)


@dataclass
class ProvenanceModel:
    variant: str = "full"
    C: float = 10.0
    temperature: float = 1.0
    policy: AbstainPolicy = field(default_factory=AbstainPolicy)
    labels: list[str] = field(default_factory=list)
    _feats: FeatureUnion | None = None
    _clf: LogisticRegression | None = None

    def fit(self, train: list[Passage]) -> ProvenanceModel:
        if len({p.label for p in train}) < 2:
            raise ValueError("need at least two models in the training data")
        self.labels = sorted({p.label for p in train})
        self._feats = build_features(self.variant)
        X = self._feats.fit_transform([p.text for p in train])  # noqa: N806
        self._clf = LogisticRegression(C=self.C, max_iter=3000)
        self._clf.fit(X, [p.label for p in train])
        assert list(self._clf.classes_) == self.labels
        return self

    def logits(self, texts: list[str]) -> np.ndarray:
        if self._clf is None or self._feats is None:
            raise RuntimeError("model is not fitted")
        z = self._clf.decision_function(self._feats.transform(texts))
        if z.ndim == 1:  # two classes: sklearn returns one score; softmax([0, z]) == sigmoid(z)
            z = np.column_stack([np.zeros_like(z), z])
        return z

    def predict_proba(self, texts: list[str], calibrated: bool = True) -> np.ndarray:
        return softmax(self.logits(texts), self.temperature if calibrated else 1.0)

    def calibrate(self, val: list[Passage], target_precision: float = 0.95) -> ProvenanceModel:
        """Fit temperature, then abstain thresholds, on the validation split only."""
        y = np.array([self.labels.index(p.label) for p in val])
        lg = self.logits([p.text for p in val])
        self.temperature = fit_temperature(lg, y)
        self.policy = choose_thresholds(softmax(lg, self.temperature), y, target_precision,
                                        self.policy.min_words)
        return self

    def attribute(self, texts: list[str]) -> list[dict]:
        probs = self.predict_proba(texts)
        out = []
        for t, p in zip(texts, probs, strict=True):
            label, reason = self.policy.decide(p, self.labels, len(t.split()))
            out.append({"label": label, "reason": reason, "top_prob": float(p.max()),
                        "scores": {lab: round(float(x), 4) for lab, x in zip(self.labels, p, strict=True)}})
        return out

    def top_features(self, label: str, k: int = 8) -> list[str]:
        """Strongest positive features for one model, for explanations."""
        names = self._feats.get_feature_names_out()
        coef = self._clf.coef_[self.labels.index(label)]
        return [str(names[i]) for i in np.argsort(coef)[::-1][:k]]
