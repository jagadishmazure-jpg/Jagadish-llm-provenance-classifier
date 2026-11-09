#!/usr/bin/env bash
# Download the five Apache-2.0 GGUF models (about 3.7 GB) into ./models and verify SHA-256.
# Only needed to regenerate the corpus; tests and the eval use the committed JSONL files.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
python3 - <<'PY' | while read -r file url sha; do
import json
d = json.load(open("scripts/models.json"))
for m in d["known"] + d["unknown"]:
    print(m["file"], m["url"], m["sha256"])
PY
  [[ -f "models/$file" ]] || curl -fL --retry 3 -o "models/$file" "$url"
  echo "$sha  models/$file" | sha256sum -c -
done
