"""Load and validate the generated corpus (JSONL), with a SHA-256 manifest for lineage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
REQUIRED = {"id": str, "label": str, "prompt_id": str, "topic_id": int, "task": str, "split": str, "text": str}
SPLITS = {"train", "val", "test"}


class CorpusError(ValueError):
    """A record failed validation. The message names the field, never the passage text."""


@dataclass(frozen=True)
class Passage:
    id: str
    label: str
    prompt_id: str
    topic_id: int
    task: str
    split: str
    text: str
    extra: tuple = ()

    def get(self, key: str, default=None):
        return dict(self.extra).get(key, default)


def validate(rec: dict) -> Passage:
    for key, typ in REQUIRED.items():
        if key not in rec:
            raise CorpusError(f"record {rec.get('id', '?')}: missing field '{key}'")
        val = rec[key]
        if not isinstance(val, typ) or isinstance(val, bool):
            raise CorpusError(f"record {rec.get('id', '?')}: field '{key}' has wrong type")
        if isinstance(val, float) and math.isnan(val):
            raise CorpusError(f"record {rec['id']}: NaN in '{key}'")
    if rec["split"] not in SPLITS:
        raise CorpusError(f"record {rec['id']}: unknown split")
    if not rec["text"].strip():
        raise CorpusError(f"record {rec['id']}: empty text")
    extra = tuple(sorted((k, v) for k, v in rec.items() if k not in REQUIRED and isinstance(v, (str, int, float))))
    return Passage(**{k: rec[k] for k in REQUIRED}, extra=extra)


def load(name: str, corpus_dir: Path = CORPUS_DIR) -> list[Passage]:
    path = corpus_dir / f"{name}.jsonl"
    out, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise CorpusError(f"{path.name}:{n}: not valid JSON") from e
        p = validate(rec)
        if p.id in seen:
            raise CorpusError(f"{path.name}:{n}: duplicate id {p.id}")
        seen.add(p.id)
        out.append(p)
    return out


def split(passages: list[Passage], which: str) -> list[Passage]:
    return [p for p in passages if p.split == which]


def check_no_leakage(passages: list[Passage]) -> None:
    """A topic must sit in exactly one split, otherwise test scores are inflated."""
    where: dict[int, set[str]] = {}
    for p in passages:
        where.setdefault(p.topic_id, set()).add(p.split)
    bad = sorted(t for t, s in where.items() if len(s) > 1)
    if bad:
        raise CorpusError(f"topics in more than one split: {bad}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(corpus_dir: Path = CORPUS_DIR) -> dict[str, dict]:
    out = {}
    for f in sorted(corpus_dir.glob("*.jsonl")):
        out[f.name] = {"sha256": sha256_file(f), "records": sum(1 for x in f.read_text().splitlines() if x.strip())}
    return out
