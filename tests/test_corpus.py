import json
from collections import Counter

import pytest

from provenance.corpus import CORPUS_DIR, CorpusError, check_no_leakage, load, manifest, split, validate

GOOD = {"id": "m:p00-letter", "label": "m", "prompt_id": "p00-letter", "topic_id": 0, "task": "letter",
        "split": "train", "text": "Dear customer, hello."}


def test_validate_good_record():
    assert validate(GOOD).label == "m"


def test_missing_fields_rejected():
    for field in ["id", "label", "text", "split", "topic_id"]:
        rec = {k: v for k, v in GOOD.items() if k != field}
        with pytest.raises(CorpusError, match=field):
            validate(rec)


def test_wrong_type_and_bool_rejected():
    with pytest.raises(CorpusError):
        validate({**GOOD, "topic_id": "0"})
    with pytest.raises(CorpusError):
        validate({**GOOD, "topic_id": True})


def test_bad_split_and_empty_text_rejected():
    with pytest.raises(CorpusError):
        validate({**GOOD, "split": "holdout"})
    with pytest.raises(CorpusError):
        validate({**GOOD, "text": "   "})


def test_errors_never_echo_passage_text(tmp_path):
    secret = "Policy number QA-99812 belongs to a customer"
    (tmp_path / "x.jsonl").write_text(json.dumps({**GOOD, "text": secret, "split": "nope"}) + "\n")
    with pytest.raises(CorpusError) as e:
        load("x", tmp_path)
    assert secret not in str(e.value)


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "x.jsonl").write_text((json.dumps(GOOD) + "\n") * 2)
    with pytest.raises(CorpusError, match="duplicate"):
        load("x", tmp_path)


def test_leakage_detected():
    a = validate(GOOD)
    b = validate({**GOOD, "id": "m:2", "split": "test"})
    with pytest.raises(CorpusError, match="more than one split"):
        check_no_leakage([a, b])


def test_real_corpus_is_complete_and_balanced():
    data = load("main")
    c = Counter(p.label for p in data)
    assert len(c) == 4 and set(c.values()) == {160}
    for lab in c:
        assert {p.prompt_id for p in data if p.label == lab} == {p.prompt_id for p in data if p.label == next(iter(c))}
    check_no_leakage(data)
    assert len(split(data, "test")) == 160


def test_unknown_model_is_out_of_set_and_test_only():
    known = {p.label for p in load("main")}
    unk = load("unknown")
    assert {p.label for p in unk}.isdisjoint(known)
    assert {p.split for p in unk} == {"test"}


def test_attack_sets_point_at_real_sources():
    ids = {p.id for p in load("main")}
    para = load("paraphrase")
    assert all(p.get("source_id") in ids for p in para)
    mim = load("mimic")
    assert all(p.get("mimicked") != p.label for p in mim)


def test_manifest_hashes_and_counts():
    m = manifest()
    assert set(m) >= {"main.jsonl", "unknown.jsonl", "paraphrase.jsonl", "mimic.jsonl"}
    assert m["main.jsonl"]["records"] == 640
    assert all(len(v["sha256"]) == 64 for v in m.values())
    assert CORPUS_DIR.name == "corpus"
