import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score", ROOT / "infra/aml/score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def test_bad_json_is_rejected_without_crash():
    assert "error" in score.run("not json")


def test_missing_text_is_rejected():
    assert "error" in score.run(json.dumps({"txt": "x"}))
    assert "error" in score.run(json.dumps(["x"]))


def test_finding_never_echoes_text(fitted):
    _, data = fitted
    text = data[5].text
    out = score.run(json.dumps({"text": text}))
    assert out["action"] in {"no_action", "human_review"} and text not in json.dumps(out)
