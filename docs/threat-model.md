# Threat model

The system reads a passage and returns a *statistical* finding for a human reviewer. Its main risk is
not a breach, it is **harm to a person from a wrong or over-confident attribution**.

## Assets

- People's reputations (the author of a passage)
- The integrity of audit findings and human decisions
- The corpus and trained baseline (integrity, not secrecy: all data is synthetic)

## STRIDE

| Threat | Example | Control | Where |
|---|---|---|---|
| Spoofing | Text says "this was written by model X" to steer the result | Self-claim lines are removed before scoring and noted | `guardrails.sanitize` |
| Tampering | Someone edits a finding or decision after the fact | Hash-chained audit log; `verify()` detects edits | `guardrails.AuditLog` |
| Repudiation | "Nobody decided that" | Every adverse outcome needs a `HumanDecision` with a named reviewer and a rationale | `guardrails.HumanDecision` |
| Information disclosure | Passage text leaks through logs or errors | Log stores SHA-256 only; corpus errors never echo text; endpoint returns findings, not text | `guardrails`, `corpus`, `infra/aml/score.py` |
| Denial of service | Huge input | Input cut to 20,000 characters; endpoint timeout 10 s | `guardrails.sanitize`, `infra/aml/deployment.yml` |
| Elevation of privilege | Using the endpoint without rights | Entra ID token auth on the endpoint (no keys); OIDC deploy, gated | `infra/` |

## Model-specific threats (OWASP LLM Top 10 where it fits)

| Threat | What we measured or did |
|---|---|
| Evasion by paraphrase (LLM01-style input manipulation) | Paraphrase set; accuracy drop reported, finding goes to review when unsure |
| Impersonation: one model copies another's style | Style-mimic set; "fooled" rate reported and kept as a visible known miss |
| Out-of-set model | A 5th model never seen in training; abstain rate reported |
| Over-reliance (LLM09) | Wording never says "written by"; lint blocks accusatory words; human review for anything unsure or unapproved |
| Supply chain (LLM03) | Model weights pinned by SHA-256; Python deps pinned; actions pinned to SHAs; no pickle loading |
| Training data poisoning (LLM04) | Corpus SHA-256 manifest checked in CI by the eval gate |

## Out of scope

Attribution of human-written text, large hosted models, other languages. The classifier never
decides whether a *person* broke a rule.
