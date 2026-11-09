"""The prompt bank used to build the corpus.

Every known model answers the *same* prompts, so a classifier cannot win by learning topics.
Splits are made by prompt id (never by passage), so test prompts are unseen during training.
The scenario is a fictional insurer, Quillmere Assurance, whose content-audit team reviews
customer-facing text that staff drafted with several approved AI assistants.
"""

from __future__ import annotations

import hashlib

TOPICS = [
    "a delayed home-insurance claim after a burst pipe",
    "how a policy excess works on a car claim",
    "travel insurance cover for a cancelled flight",
    "why a premium went up at renewal",
    "what to do after a minor car accident",
    "storm damage to a garden fence",
    "lost luggage on a business trip",
    "adding a young driver to a family car policy",
    "contents cover for a laptop stolen from a cafe",
    "pet insurance for a dog's knee surgery",
    "flood risk and buildings insurance",
    "a claim declined because of a policy exclusion",
    "switching to paperless policy documents",
    "a windscreen chip repair",
    "cover for working from home equipment",
    "a fire in a rented flat kitchen",
    "how no-claims discount protection works",
    "subsidence cracks in an older house",
    "a cyber-fraud scam targeting policyholders",
    "accidental damage to a television",
    "gadget cover for a cracked phone screen",
    "a hailstorm damaging parked cars",
    "landlord insurance and a tenant leak",
    "how long a typical claim takes to settle",
    "an online quote that differs from the final price",
    "a burglary while the family was on holiday",
    "replacing a boiler after a breakdown",
    "insuring an electric bicycle",
    "a policyholder moving house mid-term",
    "complaints and the right to escalate",
    "a frozen pipe during a cold snap",
    "a tree falling on a neighbour's roof",
    "small-business cover for a bakery",
    "a vacant property while the owner is abroad",
    "jewellery cover for an engagement ring",
    "a rental car excess waiver",
    "an elderly customer needing extra support",
    "a data-protection request about stored calls",
    "smoke-alarm discounts on home cover",
    "a claim for spoiled freezer food after a power cut",
]

TASKS = {
    "letter": "Write a short, friendly customer letter (about 80 words) from an insurer about {topic}.",
    "faq": "Write one FAQ answer (about 80 words) for an insurer's help page about {topic}.",
    "summary": "Write a brief internal case note (about 80 words) summarising {topic}.",
    "post": "Write a short social media post (about 60 words) from an insurer giving tips on {topic}.",
}


def build_prompts() -> list[dict]:
    """Return the full, ordered prompt bank (160 prompts)."""
    out = []
    for t_idx, topic in enumerate(TOPICS):
        for task, template in TASKS.items():
            pid = f"p{t_idx:02d}-{task}"
            out.append({"prompt_id": pid, "topic_id": t_idx, "task": task,
                        "prompt": template.format(topic=topic)})
    return out


def split_of(topic_id: int) -> str:
    """Deterministic split by *topic*, so a topic never appears in two splits.

    25 topics train (100 prompts), 5 validation (20 prompts), 10 test (40 prompts).
    The assignment uses a hash, not list order, so it does not follow topic themes.
    """
    rank = sorted(range(len(TOPICS)), key=lambda i: hashlib.sha256(f"split-{i}".encode()).hexdigest())
    pos = rank.index(topic_id)
    if pos < 25:
        return "train"
    if pos < 30:
        return "val"
    return "test"
