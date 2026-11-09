# Interview points

**30-second version.** I built a model-attribution baseline for AI content governance. Five small open
models answered the same 160 insurance prompts on CPU, so the classifier has to learn style, not topic.
TF-IDF plus stylometry into logistic regression gets 66.2%<!-- m:variants.full.accuracy.rate --> on unseen
topics against 25% for chance. I calibrated it, added an "unknown model" rule, attacked it with
paraphrase, truncation, style copying and an unseen model, and wrapped it in rules so it can never accuse
anyone: anything unsure or unapproved goes to a named human, logged in a hash-chained audit trail.

**Design choices I would defend**

- *Same prompts for every model, split by topic.* Otherwise you measure topic leakage, not provenance.
- *Baseline before transformers.* With 400 training passages and 160 test passages, a fine-tune would be
  hard to tell from noise. The fine-tune exists behind a flag and is honestly marked "not run".
- *Calibrate, then abstain.* Temperature 1.6 cut ECE roughly in half; the abstain rule cuts wrong names to
  5.0%<!-- m:selective.wrong_attribution.rate --> of passages, at the cost of coverage.
- *Never auto-accuse.* Attribution is a probability about a model, not a fact about a person.

**Things that went wrong and what I did**

- The 95% precision target set on 80 validation passages became 86.7% on test. I report it as a known
  gap and keep a test that asserts it, rather than tuning on test.
- Style copying fools the model on about a quarter of attacks. That is why low confidence and unapproved
  matches go to a person.
- The models wrote real brand names into their answers. A hashed scrub list replaces them, and a hashed
  brand check in CI stops new ones.
- Some passages say "I am Qwen". That is a free signal real text would not give, so I report how often it happens.

**What I would do next**

Larger and hosted models, human-edited drafts, more test passages to tighten intervals, a watermark
check where vendors support one, and private networking for the endpoint.
