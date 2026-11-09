# ADR 0003: Calibrate, then abstain

**Status:** accepted

**Context.** A raw logistic regression trained to near-zero loss is over-confident. An audit team acting
on over-confident scores could wrongly suspect a colleague.

**Decision.** Fit one temperature on the validation split. Then pick the lowest probability threshold
whose accepted validation passages reach 95% precision (margin = threshold / 4), and require 30+ words.
Anything below goes to "unknown model" and a human. The out-of-set model is never used for tuning.

**Consequences.** Coverage drops, wrong attributions drop. Both are reported with Wilson intervals.
