import pytest

from provenance import finetune


def test_finetune_off_by_default(monkeypatch):
    monkeypatch.delenv(finetune.FLAG, raising=False)
    assert not finetune.enabled()


def test_run_refuses_when_disabled(monkeypatch):
    monkeypatch.delenv(finetune.FLAG, raising=False)
    with pytest.raises(finetune.FinetuneDisabled, match="off"):
        finetune.run()


def test_flag_turns_it_on(monkeypatch):
    monkeypatch.setenv(finetune.FLAG, "1")
    assert finetune.enabled()
