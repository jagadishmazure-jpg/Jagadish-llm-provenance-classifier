"""Governance layer around the classifier.

Rules (each one has tests):
1. Never auto-accuse. The only automatic actions are `no_action` and `human_review`.
   Anything adverse needs a recorded decision by a named human reviewer.
2. Low confidence, short text or two close candidates -> `inconclusive` -> human review.
3. A confident match to an assistant that is not on the approved list -> human review,
   worded as "statistically consistent with", never "written by".
4. Report wording is linted against accusatory phrases before it leaves the system.
5. Self-claims inside the text ("this was written by ...") are removed before scoring.
6. The audit log stores a SHA-256 of the passage, never the passage, and is hash-chained.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .abstain import UNKNOWN

MAX_CHARS = 20_000
BANNED = re.compile(r"\b(cheat\w*|plagiar\w*|proof|proves?|guilty|definitely|certainly written|"
                    r"was written by|violat\w*|caught|dishonest\w*)\b", re.I)
SELF_CLAIM = re.compile(r"^.*\b(written|generated|authored|produced|drafted)\s+(by|with|using)\b.*$", re.I | re.M)
ACTIONS = {"no_action", "human_review"}
DECISIONS = {"no_issue", "coaching", "escalate_to_compliance"}


class GuardrailError(ValueError):
    pass


@dataclass(frozen=True)
class Finding:
    passage_sha256: str
    status: str               # "consistent_with" | "inconclusive"
    candidate: str | None
    confidence: float
    approved: bool | None
    action: str               # "no_action" | "human_review"
    reasons: tuple[str, ...]
    wording: str
    model_version: str


def sanitize(text: str) -> tuple[str, list[str]]:
    """Return text safe to score plus notes on what was changed."""
    if not isinstance(text, str):
        raise GuardrailError("text must be a string")
    notes = []
    if len(text) > MAX_CHARS:
        text, notes = text[:MAX_CHARS], ["cut to 20,000 characters"]
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    cleaned = SELF_CLAIM.sub("", text)
    if cleaned != text:
        notes.append("removed a line that claims who wrote the text")
    return cleaned.strip(), notes


def lint_wording(wording: str) -> str:
    hit = BANNED.search(wording)
    if hit:
        raise GuardrailError(f"report wording uses a blocked phrase: {hit.group(0)!r}")
    return wording


def review(attribution: dict, text: str, approved_models: set[str], model_version: str = "baseline-0.1") -> Finding:
    """Turn a classifier attribution into a governance finding."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    reasons = [attribution["reason"]]
    cand = attribution["label"]
    conf = round(float(attribution["top_prob"]), 3)
    if cand == UNKNOWN:
        wording = (f"Inconclusive: the passage does not match any assistant we track closely enough "
                   f"(top score {conf:.2f}). A reviewer may look at it; no conclusion has been drawn.")
        f = Finding(digest, "inconclusive", None, conf, None, "human_review", tuple(reasons), wording, model_version)
    else:
        ok = cand in approved_models
        if ok:
            action = "no_action"
            wording = (f"The passage is statistically consistent with {cand}, an approved assistant "
                       f"(score {conf:.2f}). No action needed.")
        else:
            action = "human_review"
            reasons.append("candidate is not on the approved list")
            wording = (f"The passage is statistically consistent with {cand}, which is not on the approved list "
                       f"(score {conf:.2f}). This is a statistical signal, not a finding about any person. "
                       f"A reviewer should talk to the author before drawing any conclusion.")
        f = Finding(digest, "consistent_with", cand, conf, ok, action, tuple(reasons), wording, model_version)
    lint_wording(f.wording)
    assert f.action in ACTIONS
    return f


@dataclass(frozen=True)
class HumanDecision:
    passage_sha256: str
    reviewer: str
    decision: str
    rationale: str

    def __post_init__(self):
        if not self.reviewer.strip() or self.reviewer.lower() in {"system", "auto", "classifier", "bot"}:
            raise GuardrailError("a named human reviewer is required")
        if self.decision not in DECISIONS:
            raise GuardrailError(f"decision must be one of {sorted(DECISIONS)}")
        if len(self.rationale.strip()) < 20:
            raise GuardrailError("rationale must explain the decision (20+ characters)")


@dataclass
class AuditLog:
    """Append-only, hash-chained JSONL. Each line carries the previous line's hash."""

    path: Path
    _last: str = field(default="0" * 64, init=False)

    def __post_init__(self):
        if self.path.exists():
            lines = [ln for ln in self.path.read_text().splitlines() if ln.strip()]
            if lines:
                self._last = json.loads(lines[-1])["hash"]

    def append(self, kind: str, record: dict) -> str:
        if "text" in record:
            raise GuardrailError("raw passage text must never be written to the audit log")
        body = {"kind": kind, "record": record, "prev": self._last}
        h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self.path.open("a") as fh:
            fh.write(json.dumps({**body, "hash": h}, sort_keys=True) + "\n")
        self._last = h
        return h

    def log_finding(self, f: Finding) -> str:
        return self.append("finding", asdict(f))

    def log_decision(self, d: HumanDecision) -> str:
        return self.append("decision", asdict(d))

    def verify(self) -> bool:
        prev = "0" * 64
        for line in self.path.read_text().splitlines():
            row = json.loads(line)
            body = {k: row[k] for k in ("kind", "record", "prev")}
            if row["prev"] != prev or hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != row["hash"]:
                return False
            prev = row["hash"]
        return True
