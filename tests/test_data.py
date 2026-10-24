import json

import pytest

from beamroot import train
from beamroot.data import get_loaders
from beamroot.train import parse_args, run


def test_synthetic_train_and_val_are_disjoint_and_cover_the_set():
    tr, va, te = get_loaders("synthetic", 32, seed=0, val_size=5000)
    n_tr, n_va = len(tr.dataset), len(va.dataset)
    assert 0 < n_va < n_tr and n_tr + n_va == len(te.dataset) == 256
    assert not set(tr.dataset.indices) & set(va.dataset.indices)
    assert set(tr.dataset.indices) | set(va.dataset.indices) == set(range(256))


def test_val_indices_depend_on_seed():
    a = get_loaders("synthetic", 32, seed=0)[1].dataset.indices
    assert a == get_loaders("synthetic", 32, seed=0)[1].dataset.indices
    assert a != get_loaders("synthetic", 32, seed=1)[1].dataset.indices


def test_val_size_zero_gives_empty_val():
    tr, va, _ = get_loaders("synthetic", 32, seed=0, val_size=0)
    assert len(va.dataset) == 0 and len(tr.dataset) == 256


def test_merge_reads_val_loader_not_test(tmp_path, monkeypatch):
    seen = {}
    real = train.merge_step

    def spy(model, loader, *a, **k):
        seen["n"] = len(loader.dataset)
        return real(model, loader, *a, **k)

    monkeypatch.setattr(train, "merge_step", spy)
    run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--epochs", "2", "--model", "dag",
                    "--merge", "--limit-batches", "2", "--runs-dir", str(tmp_path)]))
    assert seen["n"] == len(get_loaders("synthetic", 128, 0)[1].dataset) != 256


def test_merge_with_empty_val_errors(tmp_path):
    with pytest.raises(SystemExit, match="validation"):
        run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--model", "dag", "--merge",
                        "--val-size", "0", "--runs-dir", str(tmp_path)]))


def test_config_records_val_size(tmp_path):
    d = run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--epochs", "1",
                        "--limit-batches", "1", "--val-size", "10", "--runs-dir", str(tmp_path)]))
    cfgs = list(tmp_path.glob("*/config.json"))
    assert json.loads(cfgs[0].read_text())["val_size"] == 10
