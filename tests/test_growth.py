import json

import pytest
import torch

from beamroot.models.dag import DAGRouterModel
from beamroot.models.graph import RouterGraph


def diamond():
    """root -> a, b; a -> la, shared; b -> shared, lb."""
    g = RouterGraph()
    g.add_node("root", "router")
    g.add_node("a", "router", parents=["root"])
    g.add_node("b", "router", parents=["root"])
    g.add_node("la", "leaf", parents=["a"])
    g.add_node("shared", "leaf", parents=["a", "b"])
    g.add_node("lb", "leaf", parents=["b"])
    return g


def test_split_turns_multi_parent_leaf_into_router_with_two_new_leaves():
    g = diamond()
    new = g.split("shared")
    assert len(new) == 2 and len(set(new)) == 2
    assert g.nodes["shared"].kind == "router"
    assert g.parents("shared") == ["a", "b"]
    assert g.children("shared") == list(new)
    for n in new:
        assert g.nodes[n].kind == "leaf"
        assert g.parents(n) == ["shared"]
        assert "." not in n
    assert sorted(g.topological_order()) == sorted(g.nodes)  # still acyclic
    assert g.root == "root"


def test_split_names_never_collide_and_rejects_non_leaf():
    g = diamond()
    a = g.split("la")
    b = g.split(a[0])
    assert len(set(g.nodes)) == len(g.nodes) == 10
    assert not set(a) & set(b)
    with pytest.raises(ValueError):
        g.split("root")


def test_choose_leaf_picks_highest_loss():
    from beamroot.growth import choose_leaf
    assert choose_leaf({"leaf0": 0.2, "leaf1": 0.9, "leaf2": 0.5}) == "leaf1"


def test_leaf_losses_weight_each_leaf_nll_by_reach():
    from beamroot.growth import leaf_losses
    torch.manual_seed(0)
    model = DAGRouterModel()
    with torch.no_grad():  # root sends everything to leaf0
        model.nodes["root"].out.weight.zero_()
        model.nodes["root"].out.bias.copy_(torch.tensor([0.0, -1e4]))
    x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    losses = leaf_losses(model, [(x, y)], "cpu")
    assert set(losses) == {"leaf0", "leaf1"}
    assert losses["leaf0"] > 0
    assert losses["leaf1"] == pytest.approx(0.0, abs=1e-6)


def test_split_leaf_keeps_soft_output_close_and_builds_modules():
    from beamroot.growth import split_leaf
    torch.manual_seed(0)
    model = DAGRouterModel(graph=diamond()).eval()
    x = torch.randn(16, 1, 28, 28)
    before = model(x).exp()
    new = split_leaf(model, "shared", noise=1e-3)
    after = model(x).exp()
    assert set(model.nodes) == set(model.graph.nodes)
    assert model.nodes["shared"](model.stem(x).flatten(1)).shape == (16, 2)
    assert model.nodes[new[0]] is not model.nodes[new[1]]
    assert (after - before).abs().max() < 1e-2
    assert not torch.equal(model.nodes[new[0]].net[0].weight, model.nodes[new[1]].net[0].weight)


def test_rebuild_optimizer_covers_new_params_and_keeps_existing_state():
    from beamroot.growth import rebuild_optimizer, split_leaf
    torch.manual_seed(0)
    model = DAGRouterModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,))
    torch.nn.functional.nll_loss(model(x), y).backward()
    opt.step()
    stem_p = next(model.stem.parameters())
    old_state = {k: v.clone() for k, v in opt.state[stem_p].items()}
    split_leaf(model, "leaf0")
    opt = rebuild_optimizer(model, opt)
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_params == {id(p) for p in model.parameters()}
    assert opt.param_groups[0]["lr"] == 1e-3
    for k, v in old_state.items():
        assert torch.equal(opt.state[stem_p][k], v)


def test_grow_splits_worst_leaf_logs_event_and_stops_at_max_nodes(tmp_path):
    from beamroot.growth import grow
    torch.manual_seed(0)
    model = DAGRouterModel()
    data = [(torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,)))]
    events = [grow(model, data, "cpu", epoch, tmp_path, max_nodes=7) for epoch in (1, 2, 3)]
    assert events[2] is None
    assert len(model.graph.nodes) == 7
    lines = [json.loads(s) for s in (tmp_path / "growth.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    for line, ev in zip(lines, events):
        assert line == ev
        assert {"epoch", "split", "new_nodes", "num_nodes", "leaf_losses"} <= set(line)
        assert line["split"] == max(line["leaf_losses"], key=line["leaf_losses"].get)
        assert all(n in model.graph.nodes for n in line["new_nodes"])
    assert [ln["epoch"] for ln in lines] == [1, 2]
    snap = json.loads((tmp_path / "graphs" / "epoch_002.json").read_text())
    assert RouterGraph.from_dict(snap).edges == model.graph.edges


def test_train_cli_grows_dag_to_cap_and_grown_checkpoint_reloads(tmp_path):
    from beamroot.train import parse_args, run
    res = run(parse_args(["--dataset", "synthetic", "--model", "dag", "--device", "cpu",
                          "--epochs", "4", "--grow-every", "1", "--max-nodes", "7",
                          "--max-k", "2", "--runs-dir", str(tmp_path)]))
    run_dir = res["run_dir"]
    events = [json.loads(s) for s in (run_dir / "growth.jsonl").read_text().splitlines()]
    assert [e["num_nodes"] for e in events] == [5, 7]
    assert (run_dir / "graphs" / "epoch_000.json").exists()
    metrics = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["num_nodes"] == 7
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    model = DAGRouterModel.from_checkpoint(ckpt).eval()
    assert len(model.graph.nodes) == 7
    x = torch.randn(4, 1, 28, 28)
    model2 = DAGRouterModel.from_checkpoint(ckpt).eval()
    assert torch.allclose(model(x), model2(x))


def test_baseline_and_grow_every_zero_do_not_grow(tmp_path):
    from beamroot.train import parse_args, run
    base = ["--dataset", "synthetic", "--device", "cpu", "--epochs", "2", "--runs-dir", str(tmp_path)]
    res = run(parse_args(base + ["--model", "dag", "--grow-every", "0"]))
    assert not (res["run_dir"] / "growth.jsonl").exists()
    res = run(parse_args(base + ["--model", "baseline"]))
    assert not (res["run_dir"] / "growth.jsonl").exists()


def test_cli_grows_by_cluster_split_and_has_no_split_init_flag(tmp_path):
    from beamroot.train import parse_args, run
    base = ["--dataset", "synthetic", "--model", "dag", "--device", "cpu", "--epochs", "2",
            "--grow-every", "1", "--max-nodes", "5", "--runs-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        parse_args(base + ["--split-init", "copy"])
    res = run(parse_args(base + ["--split-noise", "0.01"]))
    ev = json.loads((res["run_dir"] / "growth.jsonl").read_text().splitlines()[0])
    assert sum(ev["cluster_sizes"]) > 0 and "fallback" in ev


def test_k_means_recovers_two_weighted_clusters():
    from beamroot.growth import k_means
    g = torch.Generator().manual_seed(0)
    a = torch.randn(40, 8, generator=g) * 0.1 + 3.0
    b = torch.randn(60, 8, generator=g) * 0.1 - 3.0
    h = torch.cat([a, b])
    w = torch.rand(100, generator=g) + 0.1
    labels = k_means(h, w, 2)
    assert labels.shape == (100,)
    assert len(set(labels[:40].tolist())) == 1 and len(set(labels[40:].tolist())) == 1
    assert labels[0] != labels[40]


def test_neuron_masks_partition_units_ties_to_first():
    from beamroot.growth import neuron_masks
    # unit 0: group 0 fires more; unit 1: group 1 fires more; unit 2: tie; unit 3: tie at 0
    h = torch.tensor([[2.0, 0.0, 1.0, 0.0],
                      [2.0, 0.0, 1.0, 0.0],
                      [0.0, 3.0, 1.0, 0.0]])
    w = torch.tensor([1.0, 0.5, 1.0])
    labels = torch.tensor([0, 0, 1])
    k0, k1 = neuron_masks(h, w, labels)
    assert k0.tolist() == [True, False, True, True]
    assert k1.tolist() == [False, True, False, False]
    assert torch.all(k0 ^ k1)


def two_cluster_loader(n=64, batch_size=32):
    """Images that are bright in the top half or the bottom half; first n/2 top."""
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 1, 28, 28, generator=g) * 0.1
    x[: n // 2, :, :14] += 1.0
    x[n // 2:, :, 14:] += 1.0
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size), x


def test_cluster_split_router_sends_each_cluster_to_its_own_leaf():
    import math
    from beamroot.growth import cluster_split
    from beamroot.losses import routing_entropy
    torch.manual_seed(0)
    model = DAGRouterModel()
    loader, x = two_cluster_loader()
    new, info = cluster_split(model, "leaf0", loader, "cpu")
    assert model.graph.children("leaf0") == list(new)
    assert sorted(info["cluster_sizes"]) == [32, 32]
    assert info["fallback"] is None and info["router_pretrain_loss"] < 0.1
    model.eval()
    with torch.no_grad():
        p = model.route(x)["log_route"]["leaf0"]
    top, bottom = p[:32].exp().mean(0), p[32:].exp().mean(0)
    assert top.max() > 0.9 and bottom.max() > 0.9 and top.argmax() != bottom.argmax()
    assert routing_entropy(p).item() < 0.25 * math.log(2)


def test_cluster_split_changes_only_new_nodes_and_leaves_keep_their_units():
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    model = DAGRouterModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    old_w = model.nodes["leaf0"].net[0].weight.clone()
    loader, _ = two_cluster_loader()
    new, _ = cluster_split(model, "leaf0", loader, "cpu", noise=1e-4)
    after = model.state_dict()
    for k, v in before.items():
        if not k.startswith("nodes.leaf0."):
            assert torch.equal(after[k], v), k
    near = [(model.nodes[n].net[0].weight - old_w).abs().amax(1) < 1e-2 for n in new]
    assert torch.all(near[0] ^ near[1])  # each unit kept by exactly one new leaf


def test_grow_cluster_logs_sizes_and_loss_and_falls_back_to_copy_when_degenerate(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    from beamroot.growth import grow
    torch.manual_seed(0)
    model = DAGRouterModel()
    loader, _ = two_cluster_loader()
    grow(model, loader, "cpu", 1, tmp_path, max_nodes=9)
    same = DataLoader(TensorDataset(torch.ones(16, 1, 28, 28), torch.zeros(16).long()),
                      batch_size=8)
    grow(model, same, "cpu", 2, tmp_path, max_nodes=9, max_k=3)
    ev1, ev2 = [json.loads(s) for s in (tmp_path / "growth.jsonl").read_text().splitlines()]
    assert sum(ev1["cluster_sizes"]) == 64
    assert ev1["fallback"] is None and ev1["router_pretrain_loss"] < 0.1
    assert ev2["cluster_sizes"] == [16, 0] and ev2["fallback"] == "copy" and ev2["k"] == 2
    assert ev2["router_pretrain_loss"] is None
    a, b = (model.nodes[n].net[0].weight for n in ev2["new_nodes"])
    assert torch.allclose(a, b, atol=1e-2)  # copy split: both near the old leaf


def test_split_k_makes_k_leaves_and_keeps_parents():
    g = diamond()
    new = g.split("shared", 3)
    assert len(new) == 3 and len(set(new)) == 3
    assert g.parents("shared") == ["a", "b"]
    assert g.children("shared") == list(new)
    assert all(g.nodes[n].kind == "leaf" and g.parents(n) == ["shared"] for n in new)


def blobs(centres, n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.cat([torch.randn(n, 8, generator=g) * 0.1 + c for c in centres])
    return h, torch.rand(len(h), generator=g) + 0.1


def test_pick_k_chooses_three_for_three_separated_blobs():
    from beamroot.growth import pick_k
    h, w = blobs([3.0, -3.0, torch.tensor([3.0, -3.0] * 4)])
    k, labels, scores = pick_k(h, w, max_k=4)
    assert k == 3
    assert set(scores) == {2, 3, 4}
    for i in range(3):
        assert len(set(labels[i * 30:(i + 1) * 30].tolist())) == 1
    assert len(set(labels.tolist())) == 3


def test_pick_k_keeps_two_for_one_blob():
    from beamroot.growth import pick_k
    h, w = blobs([0.0], n=90)
    k, labels, scores = pick_k(h, w, max_k=4)
    assert k == 2 and set(labels.tolist()) == {0, 1}


def three_cluster_loader(n=96, batch_size=32):
    """Images bright in the top, middle or bottom band; n/3 of each, in that order."""
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 1, 28, 28, generator=g) * 0.1
    third = n // 3
    for i, rows in enumerate((slice(0, 9), slice(9, 19), slice(19, 28))):
        x[i * third:(i + 1) * third, :, rows] += 1.0
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size), x


def test_cluster_split_three_way_builds_three_leaves_with_disjoint_units():
    import math
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    model = DAGRouterModel()
    old_w = model.nodes["leaf0"].net[0].weight.clone()
    loader, x = three_cluster_loader()
    new, info = cluster_split(model, "leaf0", loader, "cpu", noise=1e-4, max_k=3)
    assert info["k"] == 3 and len(new) == 3
    assert set(info["silhouettes"]) == {"2", "3"}
    assert sorted(info["cluster_sizes"]) == [32, 32, 32]
    assert math.isfinite(info["router_pretrain_loss"]) and info["fallback"] is None
    assert model.graph.children("leaf0") == list(new)
    assert model.nodes["leaf0"](model.stem(x).flatten(1)).shape == (96, 3)
    near = torch.stack([(model.nodes[n].net[0].weight - old_w).abs().amax(1) < 1e-2
                        for n in new])
    assert torch.all(near.sum(0) == 1)  # each unit kept by exactly one new leaf


def test_grow_picks_largest_k_that_fits_the_node_budget(tmp_path):
    from beamroot.growth import grow
    torch.manual_seed(0)
    model = DAGRouterModel()  # 3 nodes
    loader, _ = three_cluster_loader()
    assert grow(model, loader, "cpu", 1, tmp_path, max_nodes=4, max_k=3) is None
    ev = grow(model, loader, "cpu", 1, tmp_path, max_nodes=5, max_k=3)
    assert ev["k"] == 2 and len(ev["new_nodes"]) == 2 and len(model.graph.nodes) == 5
    assert set(ev["silhouettes"]) == {"2"}
    ev = grow(model, loader, "cpu", 2, tmp_path, max_nodes=20, max_k=3)
    assert ev["k"] == len(ev["new_nodes"]) and ev["num_nodes"] == 5 + ev["k"]


def test_cli_max_k_flag_reaches_growth(tmp_path):
    from beamroot.train import parse_args, run
    res = run(parse_args(["--dataset", "synthetic", "--model", "dag", "--device", "cpu",
                          "--epochs", "2", "--max-nodes", "9", "--max-k", "4",
                          "--runs-dir", str(tmp_path)]))
    ev = json.loads((res["run_dir"] / "growth.jsonl").read_text().splitlines()[0])
    assert set(ev["silhouettes"]) == {"2", "3", "4"}
    assert ev["k"] == len(ev["new_nodes"])


def test_cluster_split_stores_group_reach_shares_as_the_router_balance_target():
    from torch.utils.data import DataLoader, TensorDataset
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    model = DAGRouterModel()
    loader, x = two_cluster_loader()
    x[:16, :, 14:] += 1.0  # move 16 of the 32 top images to the bottom group: 16 vs 48
    x[:16, :, :14] -= 1.0
    _, info = cluster_split(model, "leaf0", loader, "cpu")
    target = model.graph.targets["leaf0"]
    assert sorted(info["cluster_sizes"]) == [16, 48]
    assert abs(sum(target) - 1) < 1e-6
    shares = [r / sum(info["cluster_reach"]) for r in info["cluster_reach"]]
    assert target == pytest.approx(shares)
    assert "root" not in model.graph.targets
    same = DataLoader(TensorDataset(torch.ones(16, 1, 28, 28), torch.zeros(16).long()),
                      batch_size=8)
    _, info = cluster_split(model, "leaf1", same, "cpu")
    assert info["fallback"] == "copy" and "leaf1" not in model.graph.targets


def test_balance_targets_survive_checkpoint_and_old_checkpoints_load_without_them():
    model = DAGRouterModel()
    model.graph.targets["root"] = [0.25, 0.75]
    ckpt = model.checkpoint()
    assert DAGRouterModel.from_checkpoint(ckpt).graph.targets == {"root": [0.25, 0.75]}
    del ckpt["graph"]["targets"]  # checkpoints from before balance targets
    assert DAGRouterModel.from_checkpoint(ckpt).graph.targets == {}


def test_removing_a_router_drops_its_balance_target():
    g = diamond()
    g.targets.update({"a": [0.3, 0.7], "b": [0.6, 0.4]})
    g.remove_node("b")
    assert g.targets == {"a": [0.3, 0.7]}


def test_class_groups_threshold_fold_and_one_class_cases():
    from beamroot.growth import class_groups
    y = torch.tensor([0] * 40 + [1] * 30 + [2] * 20 + [3] * 5 + [4] * 5)
    w = torch.ones(100)
    # 0, 1, 2 qualify (>= 10%); 3 and 4 share a group
    groups, labels = class_groups(y, w, max_k=4)
    assert groups == [[0], [1], [2], [3, 4]]
    assert labels.tolist() == [0] * 40 + [1] * 30 + [2] * 20 + [3] * 10
    # max_k 3 folds the smallest qualifying class (2) into the shared group
    groups, labels = class_groups(y, w, max_k=3)
    assert groups == [[0], [1], [2, 3, 4]]
    assert labels.tolist() == [0] * 40 + [1] * 30 + [2] * 30
    # max_k 2: biggest class vs the rest
    assert class_groups(y, w, max_k=2)[0] == [[0], [1, 2, 3, 4]]
    # reach weights decide the mass, not the counts
    w2 = torch.cat([torch.full((40,), 0.01), torch.ones(60)])
    assert class_groups(y, w2, max_k=2)[0] == [[1], [0, 2, 3, 4]]
    # one qualifying class: that class vs the rest
    y1 = torch.tensor([0] * 50 + list(range(1, 11)) * 5)
    assert class_groups(y1, torch.ones(100), max_k=4)[0] == [[0], list(range(1, 11))]
    # every class below 10%: nothing qualifies
    y2 = torch.arange(20) % 20
    assert class_groups(y2, torch.ones(20), max_k=4) is None
    # all mass in one class: nothing to split against
    assert class_groups(torch.zeros(10, dtype=torch.long), torch.ones(10), max_k=4) is None


def test_cluster_split_class_policy_gives_one_router_output_per_group():
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    model = DAGRouterModel()
    loader, x = three_cluster_loader()
    ys = torch.cat([y for _, y in loader])
    new, info = cluster_split(model, "leaf0", loader, "cpu", noise=1e-4, max_k=4,
                              policy="class")
    n_groups = len(info["groups"])
    assert info["policy"] == "class" and info["fallback"] is None
    assert info["k"] == n_groups == len(new) >= 2
    assert sorted(c for g in info["groups"] for c in g) == sorted(set(ys.tolist()))
    assert model.nodes["leaf0"](model.stem(x).flatten(1)).shape == (96, n_groups)
    assert sum(model.graph.targets["leaf0"]) == pytest.approx(1.0)
    assert info["cluster_sizes"] == [int(torch.isin(ys, torch.tensor(g)).sum())
                                     for g in info["groups"]]


def test_class_policy_falls_back_to_cluster_when_no_class_qualifies():
    from torch.utils.data import DataLoader, TensorDataset
    from beamroot.growth import cluster_split
    torch.manual_seed(0)
    model = DAGRouterModel()
    loader, x = three_cluster_loader()
    y = torch.arange(96) % 20  # 20 classes of 5%
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    new, info = cluster_split(model, "leaf0", loader, "cpu", max_k=2, policy="class")
    assert info["policy"] == "cluster" and info["policy_fallback"] == "no class >= 10% of reach"
    assert len(new) == 2 and "groups" not in info


def test_cluster_policy_event_is_unchanged_and_cli_flag_threads_to_growth(tmp_path):
    from beamroot.train import parse_args, run
    assert parse_args([]).split_policy == "cluster"
    res = run(parse_args(["--dataset", "synthetic", "--model", "dag", "--device", "cpu",
                          "--epochs", "2", "--max-nodes", "9", "--max-k", "4",
                          "--split-policy", "class", "--runs-dir", str(tmp_path)]))
    ev = json.loads((res["run_dir"] / "growth.jsonl").read_text().splitlines()[0])
    assert ev["policy"] == "class" and len(ev["groups"]) == ev["k"] == len(ev["new_nodes"])


def test_preset_sets_split_flags_and_explicit_flags_override():
    from beamroot.train import parse_args
    explainable = parse_args([])
    assert (explainable.preset, explainable.max_k, explainable.split_policy) == (
        "explainable", 4, "cluster")
    pure = parse_args(["--preset", "pure"])
    assert (pure.max_k, pure.split_policy) == (4, "class")
    binary = parse_args(["--preset", "pure", "--max-k", "2", "--split-policy", "cluster"])
    assert (binary.max_k, binary.split_policy) == (2, "cluster")
