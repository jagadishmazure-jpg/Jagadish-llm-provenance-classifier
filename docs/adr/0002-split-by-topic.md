# ADR 0002: Split by topic, same prompts for every model

**Status:** accepted

**Context.** If models answered different prompts, a classifier could learn topics instead of style.
If the same topic sat in train and test, scores would be inflated.

**Decision.** Every known model answers the same 160 prompts. Splits are by topic (25/5/10), assigned by
a hash, and checked by `corpus.check_no_leakage`.

**Consequences.** Test topics are unseen, so the score is closer to what a new batch of letters would get.
