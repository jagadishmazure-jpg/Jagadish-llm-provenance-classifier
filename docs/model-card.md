# Model card: provenance baseline 0.1

## Intended use

Give a content-audit reviewer a **statistical hint** about which of four known small assistants most
likely drafted a short English insurance-style passage, or say "unknown". Every finding is either
`no_action` (approved assistant) or `human_review`.

## Not for

- Deciding whether a person broke a rule, or any employment, disciplinary or legal decision.
- Large hosted models, other domains or languages, or text a person has rewritten.
- Proving that text was written by a human.

## Model

Character 2-5 gram and word 1-2 gram TF-IDF plus 32 stylometry features, multinomial logistic
regression (C = 10), temperature scaling (T = 1.6<!-- m:calibration.temperature -->) and an abstain rule
(top score >= 0.70<!-- m:policy.min_prob -->, margin >= 0.175<!-- m:policy.min_margin -->, 30+ words). Trains in
seconds on CPU from `data/corpus/main.jsonl`.

## Evaluation (synthetic, 160 test passages on 10 unseen topics)

| Metric | Value |
|---|---|
| Accuracy, no abstaining | 66.2%<!-- m:variants.full.accuracy.rate --> (95% CI 58.6-73.1%) |
| Majority-class baseline | 25.0%<!-- m:majority_baseline.rate --> |
| Coverage with abstaining | 37.5%<!-- m:selective.coverage.rate --> |
| Accuracy when it attributes | 86.7%<!-- m:selective.accuracy_when_attributed.rate --> |
| Wrong names, share of all passages | 5.0%<!-- m:selective.wrong_attribution.rate --> |
| Unseen model sent to "unknown" | 80.0%<!-- m:unknown_model.abstained.rate --> |
| Style-copy attacks attributed to the copied model | 26.2%<!-- m:robustness.style_mimic.fooled_into_mimicked_model.rate --> |
| ECE after calibration | 0.073<!-- m:calibration.after.ece --> |

Full report: [results/report.md](../results/report.md).

## Known limitations

See the README "Limitations and known failures". The biggest: copied style fools it, and test
precision when it attributes is below the 95% target chosen on validation.

## Transparency notes (NIST AI RMF / EU AI Act)

- **Map:** risk is reputational harm to staff from a wrong attribution. Mitigated by human review,
  non-accusatory wording and an audit trail.
- **Measure:** all metrics are reproduced in CI from committed data with hashes.
- **Manage:** thresholds and the approved-model list are code; any change is a reviewed pull request.
- **Govern:** staff should be told the audit exists. Outputs are labelled as statistical signals. If used
  in a workplace in the EU, check whether the use falls into a high-risk category before deploying.
