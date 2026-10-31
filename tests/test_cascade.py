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


def _copy_split(leaf_init, centres=(-3.0, 3.0), **kw):
    import copy
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    m = CascadeModel(start_step=2)
    leaf = m.graph.leaves()[0]
    old = copy.deepcopy(m.nodes[leaf])
    new, _ = cluster_split(m, leaf, _blobs(centres), "cpu", limit_batches=4,
                           silhouette_min=kw.pop("smin", 0.1), leaf_init=leaf_init, **kw)
    return m, old, new


def _max_diff(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))


def test_copy_init_children_match_parent_up_to_noise():
    m, old, new = _copy_split("copy")
    assert all(_max_diff(m.nodes[n], old) < 0.02 for n in new)
    assert _max_diff(m.nodes[new[0]], m.nodes[new[1]]) > 0  # noise differs per child


def test_fresh_init_children_differ_from_parent():
    m, old, new = _copy_split("fresh")
    assert all(_max_diff(m.nodes[n], old) > 0.02 for n in new)


def test_copy_init_with_bump_is_fresh():
    m, old, new = _copy_split("copy", centres=(0.0,), smin=0.99)
    assert all(m.steps[n] == 3 for n in new)
    assert all(m.nodes[n].step == 3 for n in new)


def test_copy_init_label_leaves_stay_fresh():
    import copy
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    m = CascadeModel(start_step=2, leaf="label")
    leaf = m.graph.leaves()[0]
    old = copy.deepcopy(m.nodes[leaf])
    new, _ = cluster_split(m, leaf, _blobs((-3.0, 3.0)), "cpu", limit_batches=4,
                           leaf_init="copy")
    assert all(_max_diff(m.nodes[n], old) > 0.02 for n in new)


def test_split_records_last_change_epoch():
    from beamroot.growth import grow
    import tempfile
    torch.manual_seed(0)
    m = CascadeModel(start_step=2)
    leaf = m.graph.leaves()[0]
    with tempfile.TemporaryDirectory() as d:
        ev = grow(m, _blobs((-3.0, 3.0)), "cpu", 5, d, 9, limit_batches=4)
    assert m.last_change_epoch[ev["split"]] == 5
    assert all(m.last_change_epoch[n] == 5 for n in ev["new_nodes"])


def test_checkpoint_keeps_last_change_and_loads_old():
    m = CascadeModel(start_step=2)
    m.last_change_epoch["root"] = 4
    ckpt = m.checkpoint()
    assert CascadeModel.from_checkpoint(ckpt).last_change_epoch == {"root": 4}
    del ckpt["last_change_epoch"]
    assert CascadeModel.from_checkpoint(ckpt).last_change_epoch == {}


def test_bump_cooldown_skips_young_router_allows_old():
    import math
    from beamroot.growth import size_checks
    stats = lambda r: {r: {"usage": [0.5, 0.5], "entropy": math.log(2)}}
    m = CascadeModel(start_step=2)
    r = _router(m)
    m.last_change_epoch[r] = 4
    assert size_checks(m, stats(r), 0.5, 0.1, "cpu", epoch=5, cooldown=2) == []
    assert m.steps[r] == 2
    out = size_checks(m, stats(r), 0.5, 0.1, "cpu", epoch=6, cooldown=2)
    assert out[0]["to"] == 3 and m.last_change_epoch[r] == 6


def _growth_epochs(tmp_path, extra):
    import json
    d = run(parse_args(["--dataset", "synthetic", "--model", "cascade", "--device", "cpu",
                        "--epochs", "6", "--limit-batches", "2", "--grow-every", "1",
                        "--max-nodes", "9", "--start-step", "2", "--no-report",
                        "--silhouette-min", "0.99", "--runs-dir", str(tmp_path), *extra]))["run_dir"]
    return [json.loads(l)["epoch"] for l in (d / "growth.jsonl").read_text().splitlines()]


def test_settle_epochs_blocks_late_growth_and_bumps(tmp_path):
    assert max(_growth_epochs(tmp_path / "a", [])) == 5
    assert max(_growth_epochs(tmp_path / "b", ["--settle-epochs", "3"])) == 3


def test_new_flags_run_with_copy_and_cooldown(tmp_path):
    _growth_epochs(tmp_path, ["--leaf-init", "copy", "--bump-cooldown", "2"])


@pytest.mark.parametrize("flags", [["--leaf-init", "copy"], ["--settle-epochs", "2"],
                                   ["--bump-cooldown", "1"], ["--router-lr-decay", "0.5"]])
@pytest.mark.parametrize("model", ["dag", "baseline"])
def test_non_cascade_rejects_new_flags(tmp_path, model, flags):
    with pytest.raises(SystemExit):
        run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--limit-batches", "2",
                        "--runs-dir", str(tmp_path), "--no-report", "--model", model, *flags]))


def test_checkpoint_keeps_decay_start_and_loads_old():
    m = CascadeModel(start_step=2)
    m.decay_start["root"] = 3
    ckpt = m.checkpoint()
    assert CascadeModel.from_checkpoint(ckpt).decay_start == {"root": 3}
    del ckpt["decay_start"]
    assert CascadeModel.from_checkpoint(ckpt).decay_start == {}


def _two_path_cascade():
    """root -> a, b; a and b both -> l1, l2: every leaf has ancestors root, a and b."""
    from beamroot.models.graph import RouterGraph
    g = RouterGraph()
    g.add_node("root", "router")
    for r in ("a", "b"):
        g.add_node(r, "router", parents=["root"])
    for leaf in ("l1", "l2"):
        g.add_node(leaf, "leaf", parents=["a", "b"])
    return CascadeModel(graph=g, start_step=2)


def test_split_marks_every_ancestor_once():
    from beamroot.growth import grow
    import tempfile
    torch.manual_seed(0)
    m = _two_path_cascade()
    with tempfile.TemporaryDirectory() as d:
        ev = grow(m, _blobs((-3.0, 3.0)), "cpu", 5, d, 20, limit_batches=4)
        assert m.decay_start == {"root": 5, "a": 5, "b": 5}
        ev2 = grow(m, _blobs((-3.0, 3.0)), "cpu", 7, d, 20, limit_batches=4)
    below_first = ev2["split"] in ev["new_nodes"]
    assert m.decay_start == {"root": 5, "a": 5, "b": 5, **({ev["split"]: 7} if below_first else {})}


def test_bump_clears_decay_start():
    import math
    from beamroot.growth import size_checks
    m = CascadeModel(start_step=2)
    r = _router(m)
    m.decay_start[r] = 2
    size_checks(m, {r: {"usage": [0.5, 0.5], "entropy": math.log(2)}}, 0.5, 0.1, "cpu", epoch=4)
    assert r not in m.decay_start


def test_node_param_groups_train_like_one_group():
    import copy
    from beamroot.growth import node_param_groups
    torch.manual_seed(0)
    a = CascadeModel(start_step=1)
    b = copy.deepcopy(a)
    opt_a = torch.optim.Adam(a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(node_param_groups(b, 1e-3), lr=1e-3)
    assert [g["node"] for g in opt_b.param_groups] == list(b.nodes)
    x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    for m, opt in ((a, opt_a), (b, opt_b)):
        torch.manual_seed(1)  # same dropout masks in both runs
        for _ in range(3):
            opt.zero_grad()
            F.nll_loss(m(x), y).backward()
            opt.step()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_router_lr_decay_is_cascade_only(tmp_path):
    with pytest.raises(SystemExit, match="--router-lr-decay.*only apply to --model cascade"):
        run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--limit-batches", "2",
                        "--runs-dir", str(tmp_path), "--no-report", "--model", "dag",
                        "--router-lr-decay", "0.5"]))


def test_router_lr_decay_run_saves_decay_start(tmp_path):
    d = run(parse_args(["--dataset", "synthetic", "--model", "cascade", "--device", "cpu",
                        "--epochs", "4", "--limit-batches", "2", "--grow-every", "1",
                        "--max-nodes", "9", "--start-step", "2", "--no-report",
                        "--router-lr-decay", "0.5", "--runs-dir", str(tmp_path)]))["run_dir"]
    ckpt = torch.load(d / "model.pt")
    # root: bumped at epoch 2 (cleared), then marked again by that epoch's split below it
    assert ckpt["decay_start"] == {"root": 2}


def _decisive(r, entropy_frac=0.1):
    import math
    return {r: {"usage": [0.5, 0.5], "entropy": entropy_frac * math.log(2)}}


def test_freeze_checks_freeze_a_decisive_ancestor_only():
    from beamroot.growth import freeze_checks
    m = CascadeModel(start_step=2)
    r = _router(m)
    assert freeze_checks(m, _decisive(r), 0.25, 0.1) == [] and m.frozen == set()
    m.decay_start[r] = 2
    out = freeze_checks(m, _decisive(r), 0.25, 0.1)
    assert out == [{"freeze": r, "entropy": _decisive(r)[r]["entropy"], "spread": 0.5}]
    assert m.frozen == {r}
    assert not any(p.requires_grad for p in m.nodes[r].parameters())


def test_freeze_checks_skip_router_between_freeze_and_bump_bars():
    from beamroot.growth import freeze_checks
    m = CascadeModel(start_step=2)
    r = _router(m)
    m.decay_start[r] = 2
    assert freeze_checks(m, _decisive(r, 0.4), 0.25, 0.1) == [] and m.frozen == set()
    st = {r: {"usage": [0.92, 0.08], "entropy": 0.01}}  # decisive but under the spread bar
    assert freeze_checks(m, st, 0.25, 0.1) == [] and m.frozen == set()


def test_frozen_router_keeps_params_and_adam_state_while_others_train():
    from beamroot.growth import freeze_checks, node_param_groups
    from beamroot.train import train_epoch
    torch.manual_seed(0)
    m = CascadeModel(start_step=1)
    opt = torch.optim.Adam(node_param_groups(m, 1e-2), lr=1e-2)
    batch = [(torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,)))]
    train_epoch(m, batch, opt, "cpu")  # Adam state exists, so momentum alone would move it
    m.decay_start["root"] = 1
    freeze_checks(m, _decisive("root"), 0.25, 0.1)
    root = {k: v.clone() for k, v in m.nodes["root"].state_dict().items()}
    state = {k: v.clone() for k, v in opt.state[next(m.nodes["root"].parameters())].items()}
    leaf = {k: v.clone() for k, v in m.nodes["leaf0"].state_dict().items()}
    train_epoch(m, batch, opt, "cpu")
    assert all(torch.equal(v, m.nodes["root"].state_dict()[k]) for k, v in root.items())
    assert all(p.grad is None for p in m.nodes["root"].parameters())
    assert all(torch.equal(v, opt.state[next(m.nodes["root"].parameters())][k])
               for k, v in state.items())
    assert not all(torch.equal(v, m.nodes["leaf0"].state_dict()[k]) for k, v in leaf.items())


def test_size_checks_skip_frozen_router():
    from beamroot.growth import freeze_checks, size_checks
    m = CascadeModel(start_step=2)
    r = _router(m)
    m.decay_start[r] = 2
    freeze_checks(m, _decisive(r), 0.25, 0.1)
    before = m.nodes[r]
    assert size_checks(m, _decisive(r, 1.0), 0.5, 0.1, "cpu", epoch=4) == []
    assert m.nodes[r] is before and m.steps[r] == 2 and m.frozen == {r}


def test_merge_unfreezes_the_merged_router():
    from beamroot.growth import merge_nodes
    from beamroot.models.graph import RouterGraph
    g = RouterGraph()  # root -> a, c; c -> b: a and b share no parent
    g.add_node("root", "router")
    g.add_node("a", "router", parents=["root"])
    g.add_node("c", "router", parents=["root"])
    g.add_node("b", "router", parents=["c"])
    g.add_node("l0", "leaf", parents=["c"])
    for leaf, parent in (("l1", "a"), ("l2", "a"), ("l3", "b"), ("l4", "b")):
        g.add_node(leaf, "leaf", parents=[parent])
    m = CascadeModel(graph=g, start_step=2)
    m.frozen = {"a", "b"}
    removed = merge_nodes(m, "a", "b")
    assert "b" in removed and m.frozen == {"a"}


def test_checkpoint_keeps_frozen_nodes_frozen_and_loads_old():
    m = CascadeModel(start_step=2)
    m.frozen.add("root")
    m.nodes["root"].requires_grad_(False)
    ckpt = m.checkpoint()
    loaded = CascadeModel.from_checkpoint(ckpt)
    assert loaded.frozen == {"root"}
    assert not any(p.requires_grad for p in loaded.nodes["root"].parameters())
    assert all(p.requires_grad for p in loaded.nodes["leaf0"].parameters())
    del ckpt["frozen"]
    assert CascadeModel.from_checkpoint(ckpt).frozen == set()


def _freeze_run(tmp_path, extra):
    import json
    d = run(parse_args(["--dataset", "synthetic", "--model", "cascade", "--device", "cpu",
                        "--epochs", "4", "--limit-batches", "2", "--grow-every", "1",
                        "--max-nodes", "9", "--start-step", "2", "--no-report",
                        "--runs-dir", str(tmp_path), *extra]))["run_dir"]
    events = [json.loads(l) for l in (d / "growth.jsonl").read_text().splitlines()]
    return events, torch.load(d / "model.pt")


def test_freeze_routers_off_freezes_and_logs_nothing(tmp_path):
    events, ckpt = _freeze_run(tmp_path, ["--freeze-entropy-max", "1.0", "--spread-min", "0"])
    assert not any("freeze" in e for e in events) and ckpt["frozen"] == []


def test_freeze_routers_freezes_ancestor_during_settle_epochs(tmp_path):
    # only epoch 1 grows; the bars always pass, so root (ancestor from epoch 1) freezes after 2
    events, ckpt = _freeze_run(tmp_path, ["--freeze-routers", "quality", "--settle-epochs", "3",
                                          "--freeze-entropy-max", "1.0", "--spread-min", "0"])
    freezes = [e for e in events if "freeze" in e]
    assert [(e["epoch"], e["freeze"]) for e in freezes] == [(2, "root")]
    assert set(freezes[0]) == {"epoch", "freeze", "entropy", "spread"}
    assert max(e["epoch"] for e in events if "split" in e) == 1
    assert ckpt["frozen"] == ["root"]


@pytest.mark.parametrize("flags", [["--freeze-routers", "quality"],
                                   ["--freeze-entropy-max", "0.5"]])
def test_freeze_flags_are_cascade_only(tmp_path, flags):
    with pytest.raises(SystemExit, match="--freeze-routers and --freeze-entropy-max only apply"):
        run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--limit-batches", "2",
                        "--runs-dir", str(tmp_path), "--no-report", "--model", "dag", *flags]))


def _eval_counts(m, rows=3):
    m.train()
    m.node_evals = m.rows_seen = m.trained_evals = 0
    m.route(torch.randn(rows, 1, 28, 28))
    return m.node_evals / m.rows_seen, m.trained_evals / m.rows_seen


@pytest.mark.parametrize("prune", [0.0, 1e-9])
def test_trained_evals_equal_node_evals_when_nothing_frozen(prune):
    m = CascadeModel(start_step=2)
    m.prune_reach = prune
    nodes, trained = _eval_counts(m)
    assert trained == nodes


@pytest.mark.parametrize("prune", [0.0, 1e-9])
def test_trained_evals_drop_by_one_when_root_frozen(prune):
    m = CascadeModel(start_step=2)
    m.prune_reach = prune
    nodes, _ = _eval_counts(m)
    m.nodes[m.graph.root].requires_grad_(False)
    nodes2, trained = _eval_counts(m)
    assert (nodes2, trained) == (nodes, nodes - 1)


def test_run_logs_trained_node_evals_and_freezes(tmp_path):
    import json
    d = run(parse_args(["--dataset", "synthetic", "--model", "cascade", "--device", "cpu",
                        "--epochs", "2", "--limit-batches", "2", "--no-report",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
    assert all(r["trained_node_evals"] == r["node_evals"] and r["freezes"] == 0 for r in rows)
