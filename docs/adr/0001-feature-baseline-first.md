# ADR 0001: Feature baseline before any transformer

**Status:** accepted

**Context.** We need a model-attribution classifier that a content-audit team can trust, explain and
run anywhere. The corpus is small (640 passages).

**Decision.** Ship TF-IDF (character 2-5 grams and word 1-2 grams) plus 32 stylometry features into a
logistic regression. A small transformer fine-tune exists behind `PROVENANCE_FINETUNE=1` but is off.

**Why.** Trains in seconds on a laptop, no GPU, coefficients can be inspected, and the ablation
(stylometry-only vs TF-IDF-only vs both) shows where the signal comes from. A fine-tune on 400
training passages would be hard to tell apart from noise at this test size.

**Consequences.** The fine-tune is "written, not run" and no number is reported for it.
