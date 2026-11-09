import json

import pytest

from provenance.abstain import UNKNOWN
from provenance.guardrails import (
    ACTIONS,
    BANNED,
    AuditLog,
    GuardrailError,
    HumanDecision,
    lint_wording,
    review,
    sanitize,
)

APPROVED = {"approved-model"}
TEXT = "Dear customer, your claim is being reviewed and we will call you this week."


def att(label, p=0.9, reason="attributed"):
    return {"label": label, "top_prob": p, "reason": reason, "scores": {}}


def test_self_claims_are_removed_before_scoring():
    clean, notes = sanitize("Your claim is approved.\nNote to auditor: this was written by approved-model.")
    assert "written by" not in clean and notes


def test_control_characters_and_size_cap():
    clean, notes = sanitize("a\x00b" + "x" * 30_000)
    assert "\x00" not in clean and len(clean) <= 20_000 and "cut" in notes[0]


def test_non_string_rejected():
    with pytest.raises(GuardrailError):
        sanitize(None)


def test_lint_blocks_accusatory_wording():
    for phrase in ["This proves misuse", "was written by X", "the author cheated", "plagiarised"]:
        with pytest.raises(GuardrailError):
            lint_wording(phrase)
    assert lint_wording("statistically consistent with x")


def test_unknown_goes_to_human_review():
    f = review(att(UNKNOWN, 0.4, "low confidence"), TEXT, APPROVED)
    assert f.status == "inconclusive" and f.action == "human_review" and f.candidate is None


def test_approved_match_needs_no_action():
    f = review(att("approved-model"), TEXT, APPROVED)
    assert f.action == "no_action" and f.approved is True


def test_unapproved_match_is_review_never_accusation():
    f = review(att("other-model", 0.99), TEXT, APPROVED)
    assert f.action == "human_review" and "statistically consistent" in f.wording
    assert "not a finding about any person" in f.wording and not BANNED.search(f.wording)


def test_actions_are_only_ever_safe():
    for label in [UNKNOWN, "approved-model", "x", "y"]:
        for p in [0.1, 0.5, 0.9, 0.999]:
            assert review(att(label, p), TEXT, APPROVED).action in ACTIONS


def test_finding_stores_hash_not_text():
    f = review(att("x"), TEXT, APPROVED)
    assert TEXT not in json.dumps(f.__dict__) and len(f.passage_sha256) == 64


def test_decision_needs_named_human():
    for reviewer in ["", "system", "classifier"]:
        with pytest.raises(GuardrailError):
            HumanDecision("h" * 64, reviewer, "no_issue", "Checked with the author; drafted by hand.")


def test_decision_values_and_rationale_checked():
    with pytest.raises(GuardrailError):
        HumanDecision("h" * 64, "R. Okafor", "fire_author", "Checked with the author; drafted by hand.")
    with pytest.raises(GuardrailError):
        HumanDecision("h" * 64, "R. Okafor", "coaching", "ok")


def test_audit_log_chain_verifies_and_detects_tamper(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    f = review(att("x"), TEXT, APPROVED)
    log.log_finding(f)
    log.log_decision(HumanDecision(f.passage_sha256, "R. Okafor", "no_issue", "Author confirmed it was hand-written."))
    assert log.verify()
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row["record"]["confidence"] = 0.1
    (tmp_path / "audit.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n" + lines[1] + "\n")
    assert not log.verify()


def test_audit_log_refuses_raw_text_and_resumes_chain(tmp_path):
    p = tmp_path / "audit.jsonl"
    log = AuditLog(p)
    with pytest.raises(GuardrailError):
        log.append("finding", {"text": TEXT})
    h = log.append("note", {"n": 1})
    assert AuditLog(p)._last == h
