# Data card: synthetic provenance corpus

## What it is

Short insurance-style passages (customer letters, FAQ answers, internal case notes, social posts) for a
fictional insurer. Every passage was **generated on CPU on 2026-11-09** by one of five small open-weight
chat models through llama.cpp. Nothing was written by a person and nothing comes from a real company.

| File | Records | What is in it |
|---|---|---|
| `data/corpus/main.jsonl` | 640 | 4 known models x 160 prompts (40 topics x 4 tasks) |
| `data/corpus/unknown.jsonl` | 40 | a 5th model the classifier never trains on, test prompts only |
| `data/corpus/paraphrase.jsonl` | 160 | every known-model test passage rewritten by `qwen2.5-0.5b` |
| `data/corpus/mimic.jsonl` | 80 | model A shown a model-B passage and told to copy its style |

SHA-256 hashes and record counts are in `results/report.md` and are recomputed by CI.

## Models (all Apache-2.0, weights never committed)

| Label | Role | Weights |
|---|---|---|
| qwen2.5-0.5b | known, also the paraphraser | Qwen2.5-0.5B-Instruct, Q4_K_M GGUF |
| qwen2.5-1.5b | known | Qwen2.5-1.5B-Instruct, Q4_K_M GGUF |
| smollm2-360m | known | SmolLM2-360M-Instruct, Q8_0 GGUF |
| smollm2-1.7b | known | SmolLM2-1.7B-Instruct, Q4_K_M GGUF |
| tinyllama-1.1b | out-of-set ("unknown") | TinyLlama-1.1B-Chat-v1.0, Q4_K_M GGUF |

Download URLs and SHA-256 of each file are in `scripts/models.json`.

## How it was made

1. `provenance/prompts.py` builds 160 prompts: 40 insurance topics x 4 writing tasks.
2. Every known model answers **the same** prompts, so topic words cannot give away the model.
3. Sampling: temperature 0.7, top-p 0.95, at most 140 new tokens, a seed derived from
   SHA-256 of `model|prompt_id`. Passages that hit the token limit end mid-sentence; this is kept,
   because it is what the models did.
4. Splits are by **topic**: 25 topics train (100 prompts), 5 validation (20), 10 test (40). A topic
   never appears in two splits (checked in code and in tests).

Regenerate: `bash scripts/fetch_models.sh && python scripts/generate_corpus.py --job all`.
Outputs can differ slightly on other CPUs or llama.cpp builds; the committed files are the
measured ones.

## Known limits

- Small models, short passages, one domain, English only. Results say nothing about large hosted models.
- One prompt per item and one sampling setting. Real staff edit drafts; we only simulate light edits.
- 160 test passages: intervals are wide (Wilson 95% intervals are reported for every rate).
