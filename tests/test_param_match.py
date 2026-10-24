import json

import pytest
import torch

from beamroot.data import dataset_info
from beamroot.models.baseline import BaselineCNN, count_params, match_width
from beamroot.train import parse_args, run


@pytest.mark.parametrize("target", [60_000, 150_000, 600_000, 2_000_000])
@pytest.mark.parametrize("name", ["mnist", "cifar10"])
def test_match_width_lands_within_10_percent(name, target):
    info = dataset_info(name)
    width, hidden = match_width(target, info["channels"], info["image_size"], info["num_classes"])
    model = BaselineCNN(info["channels"], info["image_size"], info["num_classes"], width, hidden)
    assert abs(count_params(model) - target) / target <= 0.10


def test_cifar_shaped_synthetic_forward():
    info = dataset_info("synthetic_cifar")
    assert (info["channels"], info["image_size"]) == (3, 32)
    out = BaselineCNN(3, 32, 10, 8)(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def smoke(tmp_path, *extra):
    return run(parse_args(["--dataset", "synthetic_cifar", "--device", "cpu", "--epochs", "2",
                           "--limit-batches", "2", "--runs-dir", str(tmp_path), *extra]))


def test_cifar_shaped_smoke_dag_then_matched_baseline(tmp_path):
    dag = smoke(tmp_path, "--model", "dag")
    assert (dag["run_dir"] / "report.html").exists()
    html = (dag["run_dir"] / "report.html").read_text()
    assert "group" in html
    base = smoke(tmp_path, "--model", "baseline", "--match-params", str(dag["run_dir"]))
    dag_params = json.loads((dag["run_dir"] / "config.json").read_text())["params"]
    base_params = json.loads((base["run_dir"] / "config.json").read_text())["params"]
    assert abs(base_params - dag_params) / dag_params <= 0.10
