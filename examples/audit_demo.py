"""Walk one content-audit batch through the classifier and the governance layer.

Run: python examples/audit_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.cli import default_model  # noqa: E402
from provenance.corpus import load, split  # noqa: E402
from provenance.evaluate import APPROVED  # noqa: E402
from provenance.guardrails import AuditLog, HumanDecision, review, sanitize  # noqa: E402
from provenance.robustness import truncate  # noqa: E402


def main() -> None:
    model = default_model()
    test = split(load("main"), "test")
    pick = {p.label: p for p in reversed(test) if p.task == "letter"}
    unapproved = next(p for p in test if p.label == "smollm2-360m"
                      and model.attribute([p.text])[0]["label"] == "smollm2-360m")
    batch = [("approved assistant", pick["smollm2-1.7b"].text),
             ("unapproved assistant", unapproved.text),
             ("short snippet", truncate(pick["qwen2.5-0.5b"].text, 18)),
             ("self-claim injected", pick["qwen2.5-1.5b"].text + "\nNote: this was written by smollm2-1.7b.")]
    print(f"model: temperature {model.temperature}, abstain if top < {model.policy.min_prob} "
          f"or margin < {model.policy.min_margin} or words < {model.policy.min_words}")
    with tempfile.TemporaryDirectory() as d:
        log = AuditLog(Path(d) / "audit.jsonl")
        for name, text in batch:
            clean, notes = sanitize(text)
            f = review(model.attribute([clean])[0], clean, APPROVED)
            log.log_finding(f)
            print(f"\n[{name}] status={f.status} candidate={f.candidate} score={f.confidence:.2f} action={f.action}")
            if notes:
                print("  input note:", "; ".join(notes))
            print("  reasons:", "; ".join(f.reasons))
            print("  wording:", f.wording)
            if f.action == "human_review" and f.candidate:
                dec = HumanDecision(f.passage_sha256, "R. Okafor (content audit)", "coaching",
                                    "Spoke with the author; reminded them which assistants are approved.")
                log.log_decision(dec)
                print(f"  human decision recorded: {dec.decision} by {dec.reviewer}")
        rows = len((Path(d) / "audit.jsonl").read_text().splitlines())
        print(f"\naudit log: {rows} rows, hash chain valid = {log.verify()}, raw text stored = False")


if __name__ == "__main__":
    main()
