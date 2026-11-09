"""Generate the corpus with small open models on CPU (llama.cpp). Resumable.

Usage:
    pip install llama-cpp-python==0.3.36
    bash scripts/fetch_models.sh            # downloads ~3.7 GB of Apache-2.0 GGUF weights
    python scripts/generate_corpus.py --models-dir models --job all

Jobs:
    main        every known model answers all 160 prompts
    unknown     a model the classifier never sees in training answers the 40 test prompts
    paraphrase  the paraphraser rewrites every known-model test passage
    mimic       model A is shown a model-B passage (different topic) and told to copy its style
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance.prompts import build_prompts, split_of  # noqa: E402

CFG = json.loads((ROOT / "scripts/models.json").read_text())
OUT = ROOT / "data/corpus"


def seed_for(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def load(models_dir: Path, label: str):
    from llama_cpp import Llama

    spec = next(m for m in CFG["known"] + CFG["unknown"] if m["label"] == label)
    return Llama(model_path=str(models_dir / spec["file"]), n_ctx=CFG["sampling"]["n_ctx"],
                 n_threads=8, verbose=False, seed=0)


def chat(llm, text: str, seed: int) -> str:
    s = CFG["sampling"]
    r = llm.create_chat_completion(messages=[{"role": "user", "content": text}],
                                   temperature=s["temperature"], top_p=s["top_p"],
                                   max_tokens=s["max_tokens"], seed=seed)
    return r["choices"][0]["message"]["content"].strip()


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_main(models_dir: Path, labels: list[str], prompts: list[dict], path: Path) -> None:
    have = done_keys(path)
    for label in labels:
        llm = None
        for p in prompts:
            rid = f"{label}:{p['prompt_id']}"
            if rid in have:
                continue
            llm = llm or load(models_dir, label)
            t0 = time.time()
            text = chat(llm, p["prompt"], seed_for(label, p["prompt_id"]))
            append(path, {"id": rid, "label": label, "prompt_id": p["prompt_id"], "topic_id": p["topic_id"],
                          "task": p["task"], "split": split_of(p["topic_id"]), "text": text,
                          "gen_seconds": round(time.time() - t0, 2)})
        del llm


def by_id(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def run_paraphrase(models_dir: Path, main: dict[str, dict], path: Path) -> None:
    have = done_keys(path)
    who = CFG["paraphraser"]
    llm = None
    for r in main.values():
        if r["split"] != "test" or f"para:{r['id']}" in have:
            continue
        llm = llm or load(models_dir, who)
        text = chat(llm, "Rewrite the following text in your own words. Keep the meaning and length. "
                         "Reply with the rewritten text only.\n\n" + r["text"], seed_for("para", r["id"]))
        append(path, {"id": f"para:{r['id']}", "label": r["label"], "paraphraser": who, "source_id": r["id"],
                      "prompt_id": r["prompt_id"], "topic_id": r["topic_id"], "task": r["task"],
                      "split": "test", "text": text})


def run_mimic(models_dir: Path, main: dict[str, dict], prompts: list[dict], path: Path) -> None:
    have = done_keys(path)
    train_by_task = {}
    for r in main.values():
        if r["split"] == "train":
            train_by_task.setdefault((r["label"], r["task"]), []).append(r)
    for actor, target in CFG["mimic_pairs"]:
        llm = None
        for p in prompts:
            if split_of(p["topic_id"]) != "test":
                continue
            rid = f"mimic:{actor}->{target}:{p['prompt_id']}"
            if rid in have:
                continue
            pool = sorted(train_by_task[(target, p["task"])], key=lambda r: r["id"])
            ex = pool[seed_for(rid) % len(pool)]
            llm = llm or load(models_dir, actor)
            text = chat(llm, "Here is an example text. Copy its writing style exactly: sentence length, "
                             "punctuation, formatting, greetings, sign-off and tone.\n\nEXAMPLE:\n"
                             f"{ex['text']}\n\nNow, in that same style: {p['prompt']}", seed_for(rid))
            append(path, {"id": rid, "label": actor, "mimicked": target, "example_id": ex["id"],
                          "prompt_id": p["prompt_id"], "topic_id": p["topic_id"], "task": p["task"],
                          "split": "test", "text": text})
        del llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--job", choices=["main", "unknown", "paraphrase", "mimic", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts()[: a.limit or None]
    if a.job in ("main", "all"):
        run_main(a.models_dir, [m["label"] for m in CFG["known"]], prompts, OUT / "main.jsonl")
    if a.job in ("unknown", "all"):
        test = [p for p in prompts if split_of(p["topic_id"]) == "test"]
        run_main(a.models_dir, [m["label"] for m in CFG["unknown"]], test, OUT / "unknown.jsonl")
    if a.job in ("paraphrase", "all"):
        run_paraphrase(a.models_dir, by_id(OUT / "main.jsonl"), OUT / "paraphrase.jsonl")
    if a.job in ("mimic", "all"):
        run_mimic(a.models_dir, by_id(OUT / "main.jsonl"), prompts, OUT / "mimic.jsonl")
    from scrub_corpus import main as scrub  # replace real consumer brand names the models wrote

    scrub()


if __name__ == "__main__":
    main()
