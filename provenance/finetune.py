"""Optional small transformer fine-tune. OFF by default, written but not run in this repo.

Enable with `PROVENANCE_FINETUNE=1 python -m provenance.finetune` after
`pip install torch transformers` (not in requirements; CI never installs them).
It fine-tunes a small encoder for sequence classification on the train split, picks the
epoch on validation, and writes test predictions to results/finetune_predictions.json so
scripts/run_eval.py can compare it with the baseline. Nothing in this repo reports a
fine-tune number, because none has been run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .corpus import ROOT, load, split

FLAG = "PROVENANCE_FINETUNE"
DEFAULT_BASE = "distilroberta-base"


class FinetuneDisabled(RuntimeError):
    pass


def enabled() -> bool:
    return os.environ.get(FLAG, "0") == "1"


def require_enabled() -> None:
    if not enabled():
        raise FinetuneDisabled(f"fine-tuning is off; set {FLAG}=1 to run it (needs torch + transformers)")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the machine
        raise FinetuneDisabled("install torch and transformers first") from e


def run(base: str = DEFAULT_BASE, epochs: int = 3, out: Path = ROOT / "results/finetune_predictions.json") -> None:
    require_enabled()
    import numpy as np  # pragma: no cover
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    data = load("main")
    tr, va, te = split(data, "train"), split(data, "val"), split(data, "test")
    labels = sorted({p.label for p in tr})
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

    def enc(ps):
        b = tok([p.text for p in ps], truncation=True, max_length=256, padding=True, return_tensors="pt")
        b["labels"] = torch.tensor([labels.index(p.label) for p in ps])
        return b

    def acc(ps):
        model.eval()
        with torch.no_grad():
            logits = model(**{k: v for k, v in enc(ps).items() if k != "labels"}).logits
        return float((logits.argmax(-1).numpy() == np.array([labels.index(p.label) for p in ps])).mean()), logits

    best, best_state = -1.0, None
    torch.manual_seed(0)
    for _ in range(epochs):
        model.train()
        for i in range(0, len(tr), 16):
            loss = model(**enc(tr[i:i + 16])).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        a, _ = acc(va)
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, logits = acc(te)
    out.write_text(json.dumps({"base": base, "val_accuracy": best, "labels": labels,
                               "pred": [labels[i] for i in logits.argmax(-1).tolist()]}, indent=1))


if __name__ == "__main__":
    try:
        run()
    except FinetuneDisabled as e:
        print(e)
        sys.exit(2)
