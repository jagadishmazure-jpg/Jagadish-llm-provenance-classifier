# ADR 0005: Retrain at start-up instead of loading a pickle

**Status:** accepted

**Context.** scikit-learn models are usually saved with pickle, and loading a pickle can run code.

**Decision.** The CLI and the Azure ML scoring script retrain the baseline from the committed corpus when
they start (a few seconds). No model file is loaded.

**Consequences.** Slightly slower start; the deployed model is always exactly what CI evaluated.
