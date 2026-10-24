import json
import math

import torch

from beamroot.losses import balance_loss, router_usage, routing_entropy
from beamroot.train import parse_args, run


def _route(p, reach=None):
    p = torch.tensor(p)
    reach = torch.zeros(len(p)) if reach is None else torch.tensor(reach).log()
    return {"r": reach}, {"r": p.log()}


def test_zero_for_uniform_usage():
    lr, lrt = _route([[0.5, 0.5], [0.5, 0.5]])
    assert balance_loss(lr, lrt).item() == 0.0 or abs(balance_loss(lr, lrt).item()) < 1e-6


def test_grows_as_usage_skews():
    losses = [balance_loss(*_route([[p, 1 - p]] * 4)).item() for p in (0.5, 0.7, 0.9, 0.99)]
    assert losses == sorted(losses) and losses[-1] > losses[1] > 1e-3


def test_unreachable_router_contributes_nothing():
    lr, lrt = _route([[0.99, 0.01]] * 4, reach=[1e-12] * 4)
    assert balance_loss(lr, lrt).item() < 1e-9


def test_usage_is_reach_weighted():
    lr, lrt = _route([[1.0, 0.0], [0.0, 1.0]], reach=[3.0, 1.0])
    u = router_usage(lr["r"], lrt["r"].clamp(min=-30))
    assert torch.allclose(u, torch.tensor([0.75, 0.25]), atol=1e-4)


def test_entropy_uniform_is_log_k():
    _, lrt = _route([[0.5, 0.5]] * 2)
    assert abs(routing_entropy(lrt["r"]).item() - math.log(2)) < 1e-6


def test_stats_in_metrics_and_weight_zero_omits_loss(tmp_path):
    base = ["--dataset", "synthetic", "--model", "dag", "--device", "cpu", "--epochs", "1",
            "--limit-batches", "2", "--runs-dir", str(tmp_path)]
    res = run(parse_args(base))
    rec = json.loads((res["run_dir"] / "metrics.jsonl").read_text().splitlines()[-1])
    assert set(rec["routing"]["root"]) == {"usage", "entropy"}
    assert len(rec["routing"]["root"]["usage"]) == 2
    res0 = run(parse_args(base + ["--balance-weight", "0"]))
    assert res0["loss"] > 0
