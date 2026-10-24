import json

from beamroot.train import parse_args, run


def tiny(tmp_path, *extra):
    return run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--epochs", "1",
                           "--limit-batches", "2", "--runs-dir", str(tmp_path), *extra]))


def test_run_dir_contents(tmp_path):
    d = tiny(tmp_path, "--name", "t")["run_dir"]
    assert d.name.endswith("-t")
    assert json.loads((d / "config.json").read_text())["seed"] == 0
    assert len((d / "metrics.jsonl").read_text().splitlines()) == 1
    assert (d / "model.pt").exists()


def test_seed_reproducible(tmp_path):
    a = tiny(tmp_path, "--seed", "3")["first_loss"]
    b = tiny(tmp_path, "--seed", "3")["first_loss"]
    assert a == b
