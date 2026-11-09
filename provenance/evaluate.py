"""End-to-end evaluation. Every README number comes from `evaluate()` via scripts/run_eval.py."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

from . import robustness as rb
from .abstain import UNKNOWN
from .corpus import CORPUS_DIR, check_no_leakage, load, manifest, split
from .guardrails import ACTIONS, BANNED, review, sanitize
from .metrics import brier, confusion, ece, macro_f1, nll, wilson
from .model import VARIANTS, ProvenanceModel

APPROVED = {"qwen2.5-1.5b", "smollm2-1.7b"}  # Quillmere Assurance's approved assistants (fictional policy)


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def _attrib_stats(model: ProvenanceModel, texts: list[str], truth: list[str], extra_label: dict | None = None) -> dict:
    raw = [model.labels[i] for i in model.predict_proba(texts).argmax(axis=1)]
    att = model.attribute(texts)
    lab = [a["label"] for a in att]
    n = len(texts)
    out = {"raw_accuracy": _rate(sum(r == t for r, t in zip(raw, truth, strict=True)), n),
           "abstain": _rate(sum(x == UNKNOWN for x in lab), n),
           "wrong_attribution": _rate(sum(x not in (UNKNOWN, t) for x, t in zip(lab, truth, strict=True)), n)}
    if extra_label:
        for key, targets in extra_label.items():
            out[key] = _rate(sum(x == g for x, g in zip(lab, targets, strict=True)), n)
    return out


def evaluate(corpus_dir: Path = CORPUS_DIR) -> dict:
    main = load("main", corpus_dir)
    check_no_leakage(main)
    tr, va, te = split(main, "train"), split(main, "val"), split(main, "test")
    res: dict = {"data": {"manifest": manifest(corpus_dir), "train": len(tr), "val": len(va), "test": len(te),
                          "models": sorted({p.label for p in main})}}
    y_te = [p.label for p in te]
    texts = [p.text for p in te]
    majority = Counter(p.label for p in tr).most_common(1)[0][0]
    res["majority_baseline"] = _rate(sum(y == majority for y in y_te), len(te))

    variants = {}
    models = {}
    for v in VARIANTS:
        m = ProvenanceModel(variant=v).fit(tr)
        pred = [m.labels[i] for i in m.predict_proba(texts, calibrated=False).argmax(axis=1)]
        variants[v] = {"accuracy": _rate(sum(a == b for a, b in zip(pred, y_te, strict=True)), len(te)),
                       "macro_f1": round(macro_f1(y_te, pred, m.labels), 4)}
        models[v] = m
    res["variants"] = variants

    m = models["full"]
    labels = m.labels
    pred = [labels[i] for i in m.predict_proba(texts, calibrated=False).argmax(axis=1)]
    res["confusion"] = {"labels": labels, "matrix": confusion(y_te, pred, labels)}
    res["per_model_recall"] = {lab: _rate(sum(p == lab for p, y in zip(pred, y_te, strict=True) if y == lab),
                                          sum(y == lab for y in y_te)) for lab in labels}
    yi = np.array([labels.index(y) for y in y_te])
    before = m.predict_proba(texts, calibrated=False)
    m.calibrate(va)
    after = m.predict_proba(texts)
    res["calibration"] = {"temperature": m.temperature,
                          "before": {"ece": round(ece(before, yi), 4), "brier": round(brier(before, yi), 4),
                                     "nll": round(nll(before, yi), 4)},
                          "after": {"ece": round(ece(after, yi), 4), "brier": round(brier(after, yi), 4),
                                    "nll": round(nll(after, yi), 4)}}
    res["policy"] = {"min_prob": m.policy.min_prob, "min_margin": m.policy.min_margin,
                     "min_words": m.policy.min_words, "target_val_precision": 0.95}
    att = m.attribute(texts)
    kept = [(a["label"], y) for a, y in zip(att, y_te, strict=True) if a["label"] != UNKNOWN]
    res["selective"] = {"coverage": _rate(len(kept), len(te)),
                        "accuracy_when_attributed": _rate(sum(a == y for a, y in kept), len(kept)) if kept else None,
                        "wrong_attribution": _rate(sum(a != y for a, y in kept), len(te))}

    rob = {}
    for name, fn in [("truncate_25_words", lambda t: rb.truncate(t, 25)),
                     ("truncate_50_words", lambda t: rb.truncate(t, 50)),
                     ("strip_formatting", rb.strip_format),
                     ("drop_greeting_signoff", rb.drop_greeting_and_signoff)]:
        rob[name] = _attrib_stats(m, [fn(t) or t for t in texts], y_te)
    findings_texts = list(texts)
    para_path = corpus_dir / "paraphrase.jsonl"
    if para_path.exists():
        para = load("paraphrase", corpus_dir)
        ptxt = [p.text for p in para]
        rob["paraphrase"] = _attrib_stats(m, ptxt, [p.label for p in para],
                                          {"attributed_to_paraphraser": [p.get("paraphraser") for p in para]})
        findings_texts += ptxt
    mimic_path = corpus_dir / "mimic.jsonl"
    if mimic_path.exists():
        mim = load("mimic", corpus_dir)
        mtxt = [p.text for p in mim]
        rob["style_mimic"] = _attrib_stats(m, mtxt, [p.label for p in mim],
                                           {"fooled_into_mimicked_model": [p.get("mimicked") for p in mim]})
        rob["style_mimic"]["pairs"] = sorted({f"{p.label}->{p.get('mimicked')}" for p in mim})
        findings_texts += mtxt
    res["robustness"] = rob

    unk_path = corpus_dir / "unknown.jsonl"
    if unk_path.exists():
        unk = load("unknown", corpus_dir)
        ua = m.attribute([p.text for p in unk])
        res["unknown_model"] = {"model": sorted({p.label for p in unk}),
                                "abstained": _rate(sum(a["label"] == UNKNOWN for a in ua), len(unk)),
                                "forced_into": dict(Counter(a["label"] for a in ua if a["label"] != UNKNOWN))}
        findings_texts += [p.text for p in unk]

    # Governance gate: every finding must be non-accusatory and low confidence must go to a human.
    fs = []
    for t in findings_texts:
        clean, _ = sanitize(t)
        fs.append(review(m.attribute([clean])[0], clean, APPROVED))
    n = len(fs)
    res["guardrails"] = {
        "findings": n,
        "auto_accusations": sum(f.action not in ACTIONS for f in fs),
        "wording_lint_pass_rate": round(sum(not BANNED.search(f.wording) for f in fs) / n, 4),
        "inconclusive_sent_to_review_rate": round(
            sum(f.action == "human_review" for f in fs if f.status == "inconclusive")
            / max(1, sum(f.status == "inconclusive" for f in fs)), 4),
        "human_review_share": round(sum(f.action == "human_review" for f in fs) / n, 4),
    }
    # Some passages name the model or its maker ("I am Qwen"): a real, easy provenance signal.
    selfm = re.compile(r"\b(qwen|smollm\w*|hugging ?face|tinyllama|alibaba)\b", re.I)
    res["self_mentions"] = {lab: sum(bool(selfm.search(p.text)) for p in main if p.label == lab) for lab in labels}
    res["top_style_features"] = {lab: [f for f in m.top_features(lab, 40) if f.startswith("style__")][:4]
                                 for lab in labels}
    return res
