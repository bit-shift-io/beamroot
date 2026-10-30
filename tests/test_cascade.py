import pytest
import torch
import torch.nn.functional as F

from beamroot.beam import beam_search
from beamroot.evaluate import evaluate_modes, load_run
from beamroot.models.cascade import CascadeModel, NodeModel, size_ladder
from beamroot.train import parse_args, run


@pytest.mark.parametrize("size,axis,expected", [
    (28, "resolution", [(7, 8, 128), (14, 16, 128), (21, 24, 128), (28, 32, 128)]),
    (28, "hidden", [(28, 32, 32), (28, 32, 64), (28, 32, 96), (28, 32, 128)]),
    (28, "both", [(7, 8, 32), (14, 16, 64), (21, 24, 96), (28, 32, 128)]),
    (32, "resolution", [(8, 8, 128), (16, 16, 128), (24, 24, 128), (32, 32, 128)]),
])
def test_ladder(size, axis, expected):
    assert [size_ladder(s, axis, size, 32, 128) for s in (1, 2, 3, 4)] == expected


@pytest.mark.parametrize("axis", ["resolution", "hidden", "both"])
@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("size", [28, 32])
def test_node_shapes(axis, step, size):
    x = torch.randn(3, 1, size, size)
    assert NodeModel("router", 2, step, axis, 1, size, 32)(x).shape == (3, 2)
    assert NodeModel("leaf", 10, step, axis, 1, size, 32)(x).shape == (3, 10)


def test_checkpoint_roundtrip_keeps_steps():
    m = CascadeModel(size_axis="both", start_step=2).eval()
    m.steps["leaf1"] = 3
    m.nodes["leaf1"] = m.build_node("leaf1")
    m.eval()
    back = CascadeModel.from_checkpoint(m.checkpoint()).eval()
    assert back.steps == {"root": 2, "leaf0": 2, "leaf1": 3}
    x = torch.randn(4, 1, 28, 28)
    assert torch.equal(m(x), back(x))


def test_beam_and_soft():
    m = CascadeModel(start_step=1).eval()
    x = torch.randn(4, 1, 28, 28)
    assert len(beam_search(m, x, 1)) == 4
    assert m(x).shape == (4, 10)


def test_train_evaluate_cascade(tmp_path):
    d = run(parse_args(["--dataset", "synthetic", "--model", "cascade", "--device", "cpu",
                        "--epochs", "1", "--limit-batches", "2", "--grow-every", "0",
                        "--size-axis", "resolution", "--start-step", "2", "--no-report",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    import json
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["size_axis"] == "resolution" and cfg["start_step"] == 2
    ckpt = torch.load(d / "model.pt")
    assert ckpt["kind"] == "cascade" and ckpt["steps"]["root"] == 2
    model, config = load_run(d, "cpu")
    from beamroot.data import get_loaders
    test = get_loaders("synthetic", 32, 0, 0)[2]
    rows = evaluate_modes(model, test, "cpu", (1,), limit_batches=1)
    assert [r["mode"] for r in rows] == ["beam 1", "soft"]


@pytest.mark.parametrize("extra", [["--merge"], ["--link-policy", "loss"]])
def test_cascade_rejects(tmp_path, extra):
    with pytest.raises(SystemExit):
        run(parse_args(["--dataset", "synthetic", "--model", "cascade", "--device", "cpu",
                        "--runs-dir", str(tmp_path), *extra]))


def _loader():
    from beamroot.data import get_loaders
    return get_loaders("synthetic", 32, 0, 0)[0]


def test_cascade_split_nodes_at_parent_step_and_random(tmp_path):
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    m = CascadeModel(size_axis="both", start_step=2)
    old = m.nodes["leaf_0"] if "leaf_0" in m.nodes else None
    leaf = m.graph.leaves()[0]
    old = {k: v.clone() for k, v in m.nodes[leaf].state_dict().items()}
    new, info = cluster_split(m, leaf, _loader(), "cpu", limit_batches=2, max_k=3)
    assert m.nodes[leaf].net[-1].out_features == len(new) == info["k"]
    assert all(m.steps[n] == 2 and m.nodes[n].step == 2 for n in (leaf, *new))
    assert info["router_pretrain_loss"] is not None and leaf in m.graph.targets
    for n in new:
        assert not torch.allclose(m.nodes[n].net[0].weight, old["net.0.weight"])


def test_cascade_copy_fallback_makes_random_leaves():
    from beamroot.growth import split_leaf
    torch.manual_seed(0)
    m = CascadeModel(start_step=3)
    leaf = m.graph.leaves()[0]
    w = m.nodes[leaf].net[0].weight.clone()
    new = split_leaf(m, leaf)
    assert len(new) == 2 and all(m.steps[n] == 3 for n in new)
    assert all(not torch.allclose(m.nodes[n].net[0].weight, w) for n in new)


def test_cascade_grows_to_max_nodes(tmp_path):
    import json
    d = run(parse_args(["--dataset", "synthetic", "--model", "cascade", "--device", "cpu",
                        "--epochs", "3", "--limit-batches", "2", "--grow-every", "1",
                        "--max-nodes", "9", "--start-step", "2", "--no-report",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    events = [e for e in map(json.loads, (d / "growth.jsonl").read_text().splitlines())
              if "split" in e]
    assert events
    ckpt = torch.load(d / "model.pt")
    for e in events:
        assert set(e["steps"]) == set(e["new_nodes"])
        assert all(ckpt["steps"][n] == s for n, s in e["steps"].items())
    assert len(ckpt["graph"]["nodes"]) > 3


def test_label_leaf_same_logits_and_grads():
    from beamroot.models.cascade import LabelLeaf
    leaf = LabelLeaf(10)
    out = leaf(torch.randn(5, 1, 28, 28))
    assert out.shape == (5, 10) and torch.equal(out[0], out[3])
    out[:, 2].sum().backward()
    assert leaf.logits.grad[2] == 5


def test_label_leaves_learn_class_routing():
    torch.manual_seed(0)
    from beamroot.losses import balance_loss, path_loss
    m = CascadeModel(leaf="label", start_step=4, num_classes=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    y = torch.arange(64) % 2  # classes 0 / 1, told apart by image brightness
    x = torch.randn(64, 1, 28, 28) * 0.1 + (y * 2 - 1).float()[:, None, None, None]
    for _ in range(400):
        opt.zero_grad()
        out = m.route(x)
        loss = (path_loss(out["log_reach"], out["leaf_log_probs"], y)
                + 0.1 * balance_loss(out["log_reach"], out["log_route"]))
        loss.backward()
        opt.step()
    reach = {n: m.route(x)["log_reach"][n].exp() for n in m.graph.leaves()}
    leaf_of = {c: max(reach, key=lambda n: reach[n][y == c].mean().item()) for c in (0, 1)}
    assert leaf_of[0] != leaf_of[1]
    assert (m(x).argmax(1) == y).float().mean() > 0.9


def test_beam1_answers_label_leaf_top_class():
    m = CascadeModel(leaf="label", start_step=1).eval()
    for n in m.graph.leaves():
        with torch.no_grad():
            m.nodes[n].logits[7] = 5.0
    res = beam_search(m, torch.randn(3, 1, 28, 28), 1)
    assert [r.pred for r in res] == [7, 7, 7]


def test_label_leaf_split_uses_highest_reach_parent_view():
    from beamroot.growth import _leaf_inputs
    torch.manual_seed(0)
    m = CascadeModel(start_step=1, leaf="label").eval()
    g = m.graph
    g.add_node("r2", "router", parents=["root"])
    g.add_node("shared", "leaf", parents=["root", "r2"])
    g.add_node("other", "leaf", parents=["r2"])
    m.steps.update(r2=1, shared=1, other=1)
    for n in ("root", "r2", "shared", "other"):
        m.nodes[n] = m.build_node(n)
    # root favours r2's side: root->shared weak, r2->shared strong
    with torch.no_grad():
        m.nodes["r2"].net[-1].bias.copy_(torch.tensor([5.0, 0.0]))
        m.nodes["root"].net[-1].bias.copy_(torch.tensor([0.0, 0.0, 5.0, 0.0]))
    kids = g.children("root")
    for sel, expect in ((None, "r2"),):
        feats, hidden, reach, ys = _leaf_inputs(m, "shared", _loader(), "cpu", 1)
        x = next(iter(_loader()))[0]
        assert torch.allclose(hidden[:len(x)], m.nodes[expect].hidden(x), atol=1e-5), kids


def test_label_leaf_run_and_config(tmp_path):
    import json
    d = run(parse_args(["--dataset", "synthetic", "--model", "cascade", "--device", "cpu",
                        "--epochs", "3", "--limit-batches", "2", "--grow-every", "1",
                        "--max-nodes", "9", "--start-step", "2", "--leaf", "label",
                        "--no-report", "--runs-dir", str(tmp_path)]))["run_dir"]
    assert json.loads((d / "config.json").read_text())["leaf"] == "label"
    assert (d / "growth.jsonl").read_text().strip()
    model, _ = load_run(d, "cpu")
    assert model.leaf == "label"


def test_leaf_rejected_on_dag(tmp_path):
    with pytest.raises(SystemExit):
        run(parse_args(["--dataset", "synthetic", "--model", "dag", "--device", "cpu",
                        "--leaf", "label", "--runs-dir", str(tmp_path)]))


def _forced(prune, p_small=0.01):
    """Cascade root forced to (1-p, p) routing, with a split under the first child."""
    from beamroot.growth import split_leaf  # noqa: F401
    m = CascadeModel(start_step=1)
    with torch.no_grad():
        last = m.nodes["root"].net[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor([1 - p_small, p_small]).log())
    m.prune_reach = prune
    return m.train()


def test_prune_zero_equals_unpruned():
    torch.manual_seed(0)
    m = CascadeModel(start_step=1).eval()
    x = torch.randn(6, 1, 28, 28)
    a = m.route(x)
    m.train().prune_reach = 0.0
    m.eval()
    m.prune_reach = 0.0
    b = m.route(x)
    for k in a["log_reach"]:
        assert torch.equal(a["log_reach"][k], b["log_reach"][k])
    assert torch.equal(a["log_probs"], b["log_probs"])


def test_pruned_child_not_evaluated_and_losses_finite():
    from beamroot.losses import balance_loss, path_loss
    m = _forced(0.05)
    calls = []
    m.nodes["leaf1"].register_forward_hook(lambda mod, i, o: calls.append(i[0].shape[0]))
    x, y = torch.randn(8, 1, 28, 28), torch.zeros(8, dtype=torch.long)
    out = m.route(x)
    assert calls == []   # leaf1 (reach 0.01 < 0.05) saw no rows
    assert torch.isneginf(out["log_reach"]["leaf1"]).all()
    loss = path_loss(out["log_reach"], out["leaf_log_probs"], y) \
        + balance_loss(out["log_reach"], out["log_route"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.node_evals == 8 * 2   # root + leaf0 only


def test_router_gets_gradient_when_child_pruned_for_some_rows():
    from beamroot.losses import path_loss
    m = _forced(0.5, p_small=0.4)   # all rows 0.4 < 0.5 -> leaf1 pruned
    m.nodes["root"].net[-1].bias.data = torch.zeros(2)
    m.nodes["root"].net[-1].bias.data[0] = 5.0   # decisive, pruned child
    x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    out = m.route(x)
    path_loss(out["log_reach"], out["leaf_log_probs"], y).backward()
    assert m.nodes["root"].net[-1].bias.grad.abs().sum() > 0


def test_no_rows_reach_node_is_finite():
    m = _forced(0.99, p_small=0.01)
    out = m.route(torch.randn(4, 1, 28, 28))
    assert torch.isfinite(out["log_probs"]).all()


def test_prune_run_logs_evals_and_dag_rejects(tmp_path):
    import json
    base = ["--dataset", "synthetic", "--device", "cpu", "--epochs", "1", "--limit-batches", "2",
            "--grow-every", "0", "--no-report", "--runs-dir", str(tmp_path)]
    d = run(parse_args(["--model", "cascade", *base]))["run_dir"]
    row = json.loads((d / "metrics.jsonl").read_text().splitlines()[-1])
    assert row["node_evals"] > 0
    with pytest.raises(SystemExit):
        run(parse_args(["--model", "dag", "--prune-reach", "0.1", *base]))


def _blobs(centres):
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(0)
    xs = torch.cat([torch.randn(64, 1, 28, 28, generator=g) * 0.3 + c for c in centres])
    return DataLoader(TensorDataset(xs, torch.zeros(len(xs), dtype=torch.long)), batch_size=32)


def _split(centres, step=2, **kw):
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    m = CascadeModel(start_step=step)
    leaf = m.graph.leaves()[0]
    new, info = cluster_split(m, leaf, _blobs(centres), "cpu", limit_batches=4,
                              silhouette_min=kw.pop("smin", 0.1), **kw)
    return m, leaf, new, info


def test_poor_split_bumps_router_and_children():
    m, leaf, new, info = _split([0.0], smin=0.99)
    b = info["bump"]
    assert (b["check"], b["from"], b["to"]) == ("silhouette", 2, 3) and "value" in b
    assert all(m.steps[n] == 3 and m.nodes[n].step == 3 for n in (leaf, *new))


def test_separated_split_does_not_bump():
    m, leaf, new, info = _split([-3.0, 3.0])
    assert "to" not in info["bump"]
    assert all(m.steps[n] == 2 for n in (leaf, *new))


def test_bump_capped_at_max_records_at_max():
    m, leaf, new, info = _split([0.0], step=4, smin=0.99)
    assert info["bump"]["at_max"] is True and "to" not in info["bump"]
    assert all(m.steps[n] == 4 for n in (leaf, *new))


def test_silhouette_min_zero_never_bumps():
    m, leaf, new, info = _split([0.0], smin=0.0)
    assert "to" not in info["bump"] and m.steps[leaf] == 2


def test_class_policy_skips_check():
    from beamroot.growth import _size_check
    m = CascadeModel(start_step=2)
    assert _size_check(m, m.graph.leaves()[0], {}, [5, 5], 0.1) == {"check": "skipped"}


def test_empty_group_bumps_and_splits_with_random_leaves():
    from beamroot.growth import _size_check
    m = CascadeModel(start_step=2)
    leaf = m.graph.leaves()[0]
    assert _size_check(m, leaf, {"2": 0.9}, [10, 0], 0.1)["to"] == 3
    assert m.steps[leaf] == 3


def test_run_logs_bump_cascade_only_and_dag_rejects(tmp_path):
    import json
    args = ["--dataset", "synthetic", "--device", "cpu", "--limit-batches", "2",
            "--runs-dir", str(tmp_path), "--no-report", "--epochs", "2"]
    run(parse_args([*args, "--model", "cascade", "--silhouette-min", "0.99"]))
    ev = [json.loads(l) for d in tmp_path.iterdir() for l in open(d / "growth.jsonl")]
    assert ev and all("bump" in e for e in ev)
    with pytest.raises(SystemExit):
        run(parse_args([*args, "--silhouette-min", "0.5"]))


def _router(m):
    return next(n for n in m.graph.nodes if m.graph.nodes[n].kind == "router")


def test_size_checks_entropy_bump_keeps_children_and_target():
    import math
    from beamroot.growth import size_checks
    m = CascadeModel(start_step=2)
    r = _router(m)
    before = m.nodes[r]
    children = list(m.graph.children(r))
    out = size_checks(m, {r: {"usage": [0.5, 0.5], "entropy": math.log(2)}}, 0.5, 0.1, "cpu")
    assert out == [{"bump": r, "check": "entropy", "from": 2, "to": 3, "value": math.log(2)}]
    assert m.steps[r] == 3 and m.nodes[r] is not before
    assert list(m.graph.children(r)) == children


def test_size_checks_spread_bump():
    from beamroot.growth import size_checks
    m = CascadeModel(start_step=2)
    r = _router(m)
    out = size_checks(m, {r: {"usage": [0.97, 0.03], "entropy": 0.13}}, 0.5, 0.1, "cpu")
    assert out[0]["check"] == "spread" and out[0]["to"] == 3


def test_size_checks_at_max_and_disabled():
    from beamroot.growth import size_checks
    m = CascadeModel(start_step=4)
    r = _router(m)
    st = {r: {"usage": [0.5, 0.5], "entropy": 0.69}}
    assert size_checks(m, st, 0.5, 0.1, "cpu")[0]["at_max"] is True
    m2 = CascadeModel(start_step=2)
    st = {r: {"usage": [0.99, 0.01], "entropy": 0.69}}
    assert size_checks(m2, st, 1.0, 0.0, "cpu") == []


def test_size_checks_ignore_unknown_router():
    from beamroot.growth import size_checks
    m = CascadeModel(start_step=2)
    assert size_checks(m, {"nope": {"usage": [.5, .5], "entropy": 0.69}}, 0.5, 0.1, "cpu") == []


def test_run_logs_router_bumps_and_dag_rejects(tmp_path):
    import json
    args = ["--dataset", "synthetic", "--device", "cpu", "--limit-batches", "2",
            "--runs-dir", str(tmp_path), "--no-report", "--epochs", "3"]
    run(parse_args([*args, "--model", "cascade", "--entropy-max", "0.0001"]))
    ev = [json.loads(l) for d in tmp_path.iterdir() for l in open(d / "growth.jsonl")]
    assert any(e.get("check") == "entropy" for e in ev)
    with pytest.raises(SystemExit):
        run(parse_args([*args, "--spread-min", "0.2"]))


def _labelled_blobs(by_class):
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(0)
    xs = torch.cat([torch.randn(64, 1, 28, 28, generator=g) * 0.3 + c for c in (-3.0, 3.0)])
    ys = (torch.arange(128) >= 64).long() if by_class else torch.arange(128) % 2
    return DataLoader(TensorDataset(xs, ys), batch_size=32)


def _purity_split(by_class, **kw):
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    m = CascadeModel(start_step=2)
    leaf = m.graph.leaves()[0]
    _, info = cluster_split(m, leaf, _labelled_blobs(by_class), "cpu", limit_batches=4,
                            silhouette_min=0.1, **kw)
    return m, leaf, info["bump"]


def test_purity_check_bumps_when_groups_mix_classes():
    m, leaf, b = _purity_split(False, purity_gain=0.05)
    assert (b["check"], b["from"], b["to"]) == ("purity", 2, 3)
    assert b["value"] <= 0.05 and m.steps[leaf] == 3


def test_purity_check_passes_when_groups_split_by_class():
    m, leaf, b = _purity_split(True, purity_gain=0.05)
    assert "to" not in b and b["purity"]["value"] > 0.4 and m.steps[leaf] == 2


def test_purity_check_off_by_default():
    m, leaf, b = _purity_split(False)
    assert "to" not in b and "purity" not in b and m.steps[leaf] == 2


def test_silhouette_and_purity_failing_bump_once_and_log_both():
    m, leaf, b = _purity_split(False, purity_gain=0.05)
    from beamroot.growth import _size_check
    m2 = CascadeModel(start_step=2)
    leaf2 = m2.graph.leaves()[0]
    r = _size_check(m2, leaf2, {"2": 0.0}, [5, 5], 0.1, purity=(0.0, 0.05))
    assert (r["check"], r["from"], r["to"]) == ("silhouette", 2, 3)
    assert r["value"] == 0.0 and r["purity"] == {"check": "purity", "value": 0.0}
    assert m2.steps[leaf2] == 3


def test_purity_flags_rejected_on_dag(tmp_path):
    with pytest.raises(SystemExit):
        run(parse_args(["--model", "dag", "--purity-check", "--dataset", "synthetic",
                        "--device", "cpu", "--runs-dir", str(tmp_path)]))


@pytest.mark.parametrize("size,res", [(28, 21), (32, 24), (28, 14), (32, 8)])
def test_avg_resize_matches_adaptive_pool(size, res):
    from beamroot.models.cascade import _avg_resize

    x = torch.rand(2, 3, size, size)
    assert torch.allclose(_avg_resize(x, res), F.adaptive_avg_pool2d(x, res), atol=1e-6)


@pytest.mark.parametrize("image_size,channels", [(28, 1), (32, 3)])
def test_smallest_step_keeps_a_spatial_grid(image_size, channels):
    """A step-1 resolution node must not pool its tiny image down to 1x1 (16 features)."""
    node = NodeModel("leaf", 10, 1, "resolution", channels, image_size, 32)
    assert node.net[0].in_features >= 100
    assert node(torch.zeros(2, channels, image_size, image_size)).shape == (2, 10)


def test_label_leaves_start_different():
    """Identical leaves give routers zero gradient (a saddle), so init must break the symmetry."""
    from beamroot.models.cascade import LabelLeaf

    a, b = LabelLeaf(10), LabelLeaf(10)
    assert not torch.equal(a.logits, b.logits)
    assert a.logits.abs().max() > 0
