# Implementation guide (for teams adopting this)

This is how to use the pattern for your own approved-assistant audit. It assumes you can collect sample
outputs from every assistant your organisation allows.

## 1. Build your own labelled corpus

- List the assistants (models and versions) you want to tell apart. Add one you do **not** allow as an
  out-of-set check.
- Write a prompt bank that looks like your real work (letters, notes, posts). Give every model the
  **same** prompts. See `provenance/prompts.py`.
- Generate with the settings staff actually use (temperature, system prompt). Record them.
- Split by topic or by prompt, never by passage (`corpus.check_no_leakage`).
- Hash every file; keep the hashes with the results (`corpus.manifest`).

## 2. Train and measure

```bash
pip install -r requirements.txt
python scripts/run_eval.py          # writes results/metrics.json and results/report.md
```

Read the confusion matrix first. Model families (for example two sizes of the same family) are usually
the hardest pair. Look at the robustness table before you look at headline accuracy.

## 3. Set the abstain policy with your reviewers

`provenance/abstain.py` picks the threshold that reaches a target precision on validation. Agree the
target with the people who will act on findings. 95% is the default here; a stricter target sends more
passages to "unknown".

## 4. Keep the governance rules

- Keep `no_action` and `human_review` as the only automatic actions.
- Keep the wording lint and adjust `BANNED` for your language and policies.
- Record every human decision (`HumanDecision`) in the hash-chained `AuditLog`.
- Tell staff the audit exists and what it can and cannot show.

## 5. Re-test whenever a model changes

A new model version is a new class. Regenerate its passages, re-run the eval, and treat any drop in
the style-mimic or paraphrase results as a reason to raise the abstain threshold.

## 6. Optional: deploy

See [deployment.md](deployment.md). Check prices and get approval first.
