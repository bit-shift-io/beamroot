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


def _graph(reach, leaf_p):
    lr = {f"l{i}": torch.tensor(r).log() for i, r in enumerate(reach)}
    lp = {f"l{i}": torch.tensor(p).log() for i, p in enumerate(leaf_p)}
    return lr, lp


def test_path_loss_is_the_leaf_cross_entropy_for_one_leaf():
    import torch.nn.functional as F
    from beamroot.losses import path_loss
    lr, lp = _graph([[1.0, 1.0]], [[[0.7, 0.3], [0.2, 0.8]]])
    y = torch.tensor([0, 1])
    ce = F.cross_entropy(lp["l0"], y)
    assert abs(path_loss(lr, lp, y).item() - ce.item()) < 1e-6


def test_path_loss_gradient_routes_towards_the_right_leaf():
    from beamroot.losses import path_loss
    y = torch.tensor([0])
    logit = torch.zeros(1, 2, requires_grad=True)
    log_route = torch.log_softmax(logit, 1)
    lp = {"a": torch.tensor([[0.9, 0.1]]).log(), "b": torch.tensor([[0.1, 0.9]]).log()}
    lr = {"a": log_route[:, 0], "b": log_route[:, 1]}
    path_loss(lr, lp, y).backward()
    assert logit.grad[0, 0] < 0 < logit.grad[0, 1]  # descent raises the right leaf's logit


def test_cli_dag_run_logs_loss_parts(tmp_path):
    result = run(parse_args(["--model", "dag", "--dataset", "synthetic", "--device", "cpu",
                             "--epochs", "2", "--no-report", "--runs-dir", str(tmp_path)]))
    last = json.loads((result["run_dir"] / "metrics.jsonl").read_text().splitlines()[-1])
    assert "ce_loss" in last and "balance_loss" in last
    assert abs(last["loss"] - (last["ce_loss"] + last["balance_loss"])) < 1e-6


def test_dag_training_minimises_the_path_loss():
    from beamroot.losses import path_loss
    from beamroot.models.dag import DAGRouterModel
    from beamroot.train import train_epoch
    torch.manual_seed(0)
    model = DAGRouterModel()
    x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    model.train()  # same mode (and RNG draws) as train_epoch's forward pass
    torch.manual_seed(1)
    with torch.no_grad():
        out = model.route(x)
        expected = path_loss(out["log_reach"], out["leaf_log_probs"], y).item()
    parts = {}
    torch.manual_seed(1)
    train_epoch(model, [(x, y)], torch.optim.SGD(model.parameters(), lr=0.0), "cpu", parts=parts)
    assert abs(parts["ce"] - expected) < 1e-6


def test_cli_has_no_loss_flag():
    import pytest
    with pytest.raises(SystemExit):
        parse_args(["--model", "dag", "--loss", "path"])


def test_balance_is_zero_at_the_target_and_positive_elsewhere():
    lr, lrt = _route([[0.75, 0.25]] * 4)
    assert abs(balance_loss(lr, lrt, {"r": [0.75, 0.25]}).item()) < 1e-6
    assert balance_loss(lr, lrt, {"r": [0.5, 0.5]}).item() > 1e-3
    lr, lrt = _route([[0.5, 0.5]] * 4)
    assert balance_loss(lr, lrt, {"r": [0.75, 0.25]}).item() > 1e-3


def test_uniform_target_matches_no_target_and_zero_share_stays_finite():
    lr, lrt = _route([[0.9, 0.1]] * 4)
    assert abs(balance_loss(lr, lrt, {"r": [0.5, 0.5]}).item() - balance_loss(lr, lrt).item()) < 1e-6
    assert math.isfinite(balance_loss(lr, lrt, {"r": [0.0, 1.0]}).item())


def test_dag_training_balances_towards_the_graph_targets():
    from beamroot.models.dag import DAGRouterModel
    from beamroot.train import train_epoch
    torch.manual_seed(0)
    model = DAGRouterModel()
    model.graph.targets["root"] = [0.9, 0.1]
    x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    model.train()
    torch.manual_seed(1)
    with torch.no_grad():
        out = model.route(x)
        expected = balance_loss(out["log_reach"], out["log_route"], {"root": [0.9, 0.1]}).item()
    parts = {}
    torch.manual_seed(1)
    train_epoch(model, [(x, y)], torch.optim.SGD(model.parameters(), lr=0.0), "cpu",
                balance_weight=1.0, parts=parts)
    assert abs(parts["balance"] - expected) < 1e-6
