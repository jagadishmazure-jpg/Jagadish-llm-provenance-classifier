"""Checks on the committed results. Known misses are asserted on purpose so they stay visible."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_eval import gate  # noqa: E402
from scrub_corpus import scrub_text  # noqa: E402


def test_eval_gate_passes_and_beats_majority(metrics):
    assert gate(metrics) == []
    assert metrics["guardrails"]["auto_accusations"] == 0
    assert metrics["calibration"]["after"]["ece"] < metrics["calibration"]["before"]["ece"]


def test_known_miss_style_mimic_fools_the_classifier(metrics):
    # KNOWN MISS: when one model copies another's style, about a quarter of passages are attributed to the
    # model being copied. Abstaining and human review are the mitigation, not a fix.
    assert metrics["robustness"]["style_mimic"]["fooled_into_mimicked_model"]["k"] > 0


def test_known_miss_out_of_set_model_sometimes_forced_into_known(metrics):
    # KNOWN MISS: the unseen model is not always caught by the abstain rule.
    u = metrics["unknown_model"]
    assert sum(u["forced_into"].values()) == u["abstained"]["n"] - u["abstained"]["k"] > 0


def test_known_gap_test_precision_below_validation_target(metrics):
    # KNOWN GAP: thresholds tuned for 95% precision on 80 validation passages give less on test.
    s = metrics["selective"]["accuracy_when_attributed"]
    assert s["rate"] < metrics["policy"]["target_val_precision"] and s["ci95"][1] < 0.95


def test_brand_scrub_is_idempotent():
    text = "Pay by " + "Pay" + "Pal today."
    once, n = scrub_text(text)
    assert n == 1 and scrub_text(once) == (once, 0)
