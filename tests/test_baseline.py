import torch

from beamroot.models.baseline import BaselineCNN
from beamroot.train import parse_args, run


def test_forward_shape():
    out = BaselineCNN()(torch.randn(5, 1, 28, 28))
    assert out.shape == (5, 10)


def test_training_reduces_loss(tmp_path):
    args = parse_args(["--dataset", "synthetic", "--model", "baseline", "--device", "cpu",
                       "--epochs", "3", "--runs-dir", str(tmp_path)])
    res = run(args)
    assert res["loss"] < res["first_loss"]
    assert res["test_acc"] > 0.9


def test_model_defaults_to_dag():
    assert parse_args(["--dataset", "synthetic"]).model == "dag"
