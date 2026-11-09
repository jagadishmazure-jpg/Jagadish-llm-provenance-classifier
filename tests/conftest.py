import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def metrics():
    return json.loads((ROOT / "results/metrics.json").read_text())


@pytest.fixture(scope="session")
def fitted():
    from provenance.corpus import load, split
    from provenance.model import ProvenanceModel

    data = load("main")
    return ProvenanceModel().fit(split(data, "train")).calibrate(split(data, "val")), data
