"""Azure ML / Microsoft Foundry scoring script for the managed online deployment. WRITTEN, NOT RUN in Azure.

init() trains the baseline from the corpus shipped inside the code asset (a few seconds, no pickle is
loaded). run() accepts {"text": "..."} and returns only the governance finding: no raw text is
returned or logged.
"""

from __future__ import annotations

import json
import logging

from provenance import cli

log = logging.getLogger("provenance.score")
_ready = False


def init() -> None:
    global _ready
    cli.default_model()
    _ready = True
    log.info("provenance model ready")


def run(raw_data: str) -> dict:
    if not _ready:
        init()
    try:
        payload = json.loads(raw_data)
    except (TypeError, json.JSONDecodeError):
        return {"error": "body must be JSON like {\"text\": \"...\"}"}
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        return {"error": "field 'text' (non-empty string) is required"}
    finding = cli.run(text)
    log.info("finding status=%s action=%s sha=%s", finding["status"], finding["action"], finding["passage_sha256"][:12])
    return finding
