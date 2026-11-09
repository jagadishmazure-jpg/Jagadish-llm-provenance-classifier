"""Replace real consumer brand names that the models wrote into their passages with invented ones.

The small models sometimes name real products or payment apps. The portfolio rule is fictional companies
only, so after generation every word whose SHA-256 is in REPLACE is swapped for an invented name
(the real names are not stored in plain text). Model and vendor self-mentions such as "I am Qwen"
are technology names and are kept on purpose: they are a real provenance signal (see docs/model-card.md).

Run: python scripts/scrub_corpus.py          (idempotent; prints how many words were replaced)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[A-Za-z]+")
REPLACE = {
    "285075a02b2679248a6b4636c3328bd3097626607c3e43c0c5498258fbfa9f29": "Solvane",
    "351de2e054024bbece8ed5720ef59b99ed956285fd3b0c077c73c3f921d57a5b": "NoteSlate",
    "3a473d8388ac5d7c0d6a627ddab3a9403efc88b48cebfff70a88ecab78f815d2": "EcoDrive",
    "865ce0cf621e0d5f9c2dccb9e5079733a83d80ea2e814d32c3300bc2c2755b7b": "Harth",
    "961a6062ba1ba2d8608e5eea8be67483d936f2f9f41e312217407198e14f087c": "Strivo",
    "9b81c0353945995b0e907e78ae0dbc56feab6c54d9a78ede3bb4667a5fa4abd9": "Nova",
    "ab8a4f6bb84d161b3622afe130be97ddfa367e1b8707e49e5437a0b87ba1ae55": "PayLantern",
    "acec4064d0fcc0d59e8178f5feda22f65c16052768b7babd91f529b129b8074d": "Corvane",
    "d19a15c70437ed6ea0e4bd1bce2cb335edaf30e47c44bb1b195937dda84ce65f": "MailNest",
    "d5b72847b212b6519602ec8eb60d20d8cfd591854cffe05dfb415a58cbca312f": "Quillpay",
    "f19d051291d4364989e4e957ef610448af3e05863ee1b812738dd8ba78704bbf": "Brenmoor",
}


def scrub_text(text: str) -> tuple[str, int]:
    n = 0

    def sub(mt: re.Match) -> str:
        nonlocal n
        new = REPLACE.get(hashlib.sha256(mt.group(0).encode()).hexdigest())
        if new is None:
            return mt.group(0)
        n += 1
        return new

    return WORD.sub(sub, text), n


def scrub_file(path: Path) -> int:
    total, rows = 0, []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        rec["text"], n = scrub_text(rec["text"])
        total += n
        rows.append(json.dumps(rec, ensure_ascii=False))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return total


def main() -> int:
    total = 0
    for f in sorted((ROOT / "data/corpus").glob("*.jsonl")):
        n = scrub_file(f)
        total += n
        print(f"{f.name}: {n} replaced")
    print(f"total replaced: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
