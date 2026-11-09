"""Command line: attribute one passage and print the governance finding as JSON.

    python -m provenance.cli --text "Dear customer, ..."
    python -m provenance.cli --file letter.txt

The baseline trains from the committed corpus at start-up (a few seconds on a laptop),
so no pickled model file is ever loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from .corpus import load, split
from .evaluate import APPROVED
from .guardrails import review, sanitize
from .model import ProvenanceModel


@lru_cache(maxsize=1)
def default_model() -> ProvenanceModel:
    data = load("main")
    return ProvenanceModel().fit(split(data, "train")).calibrate(split(data, "val"))


def run(text: str) -> dict:
    clean, notes = sanitize(text)
    model = default_model()
    att = model.attribute([clean])[0]
    finding = asdict(review(att, clean, APPROVED))
    finding["input_notes"] = notes
    finding["scores"] = att["scores"]
    return finding


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file", type=Path)
    a = ap.parse_args(argv)
    text = a.text if a.text is not None else a.file.read_text(encoding="utf-8")
    print(json.dumps(run(text), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
