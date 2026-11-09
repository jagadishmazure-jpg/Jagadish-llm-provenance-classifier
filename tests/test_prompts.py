from collections import Counter

from provenance.prompts import TASKS, TOPICS, build_prompts, split_of


def test_prompt_bank_size():
    assert len(build_prompts()) == len(TOPICS) * len(TASKS) == 160


def test_prompt_ids_unique():
    ids = [p["prompt_id"] for p in build_prompts()]
    assert len(ids) == len(set(ids))


def test_split_sizes_by_prompt():
    c = Counter(split_of(p["topic_id"]) for p in build_prompts())
    assert c == {"train": 100, "val": 20, "test": 40}


def test_every_prompt_names_its_topic():
    for p in build_prompts():
        assert TOPICS[p["topic_id"]] in p["prompt"]


def test_split_is_not_list_order():
    first_ten = {split_of(i) for i in range(10)}
    assert len(first_ten) > 1
