# ADR 0004: The classifier never accuses anyone

**Status:** accepted

**Context.** Model attribution is probabilistic and can be fooled (see the style-mimic results). Acting
on it automatically would be unfair and, in many places, a legal risk.

**Decision.** The only automatic actions are `no_action` and `human_review`. Wording says
"statistically consistent with", never "written by". A lint blocks accusatory words. Any adverse
outcome requires a `HumanDecision` with a named reviewer and a written rationale, stored in a
hash-chained audit log.

**Consequences.** More work for reviewers, by design. The governance gate in CI fails if any finding
breaks these rules.
