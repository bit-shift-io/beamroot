import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from beamroot.beam import beam_search
from beamroot.cost import model_costs, module_macs
from beamroot.evaluate import load_run
from beamroot.evaluate import main as evaluate_main
from beamroot.models.tree import TreeModel, deep4_blocks
from beamroot.train import parse_args, run


def cifar_tree(width=16):
    return TreeModel(in_channels=3, image_size=32, num_classes=10, width=width).eval()


def test_one_leaf_tree_is_the_plain_deep4_network():
    torch.manual_seed(0)
    tree = cifar_tree()
    plain = nn.Sequential(*deep4_blocks(3, 16), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                          nn.Linear(128, 10)).eval()
    leaf = tree.nodes["root"]
    weights = [p for p in leaf.segment.parameters()] + [p for p in leaf.head.parameters()]
    with torch.no_grad():
        for dst, src in zip(plain.parameters(), weights, strict=True):
            dst.copy_(src)
    x = torch.randn(4, 3, 32, 32)
    assert torch.allclose(tree(x), F.log_softmax(plain(x), dim=1), atol=1e-6)


def test_checkpoint_round_trip_through_load_run(tmp_path):
    tree = cifar_tree()
    torch.save(tree.checkpoint(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "cifar10"}))
    back, _ = load_run(tmp_path, "cpu")
    assert isinstance(back, TreeModel) and not back.training
    x = torch.randn(4, 3, 32, 32)
    assert torch.equal(tree(x), back(x))


def test_beam_one_on_a_one_node_tree_predicts_the_forward_argmax():
    torch.manual_seed(0)
    tree = cifar_tree()
    x = torch.randn(6, 3, 32, 32)
    results = beam_search(tree, x, 1)
    assert [r.pred for r in results] == tree(x).argmax(1).tolist()
    assert all(r.evaluated == ["root"] and r.paths[0].nodes == ["root"] for r in results)


def test_one_node_tree_macs_equal_the_base_network():
    tree = cifar_tree()
    plain = nn.Sequential(*deep4_blocks(3, 16), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                          nn.Linear(128, 10))
    stem, nodes = model_costs(tree)
    base = module_macs(plain, torch.zeros(1, 3, 32, 32))
    assert stem == 0 and nodes == {"root": base}
    assert round(base / 1e6, 2) == 6.34


def test_train_run_tree_writes_model_and_metrics(tmp_path):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "1", "--limit-batches", "2", "--max-nodes", "1",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    assert (d / "metrics.jsonl").read_text().strip()
    config = json.loads((d / "config.json").read_text())
    assert config["tree_growth"] == "settle" and config["plateau_delta"] == 0.03  # Q11 default
    ckpt = torch.load(d / "model.pt")
    assert ckpt["kind"] == "tree"
    model, _ = load_run(d, "cpu")
    assert list(model.nodes) == ["root"]
    rows = evaluate_main([str(d), "--device", "cpu", "--beams", "1", "--limit-batches", "1"])
    assert rows[0]["mode"] == "beam 1" and rows[0]["mean_macs"] == model_costs(model)[1]["root"]
    assert rows[0]["ms_per_image"] > 0 and (d / "eval.json").exists()


@pytest.mark.parametrize("flags", [["--merge"], ["--link-policy", "loss"],
                                   ["--leaf", "label"], ["--settle-epochs", "1"],
                                   ["--size-axis", "hidden"], ["--start-step", "2"]])
def test_dag_and_cascade_only_flags_exit_for_tree(tmp_path, flags):
    with pytest.raises(SystemExit, match="tree|cascade"):
        run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "1", "--limit-batches", "1",
                        "--runs-dir", str(tmp_path), *flags]))
    assert not list(tmp_path.iterdir())


def test_split_children_copy_the_old_leaf_and_router_gives_two_way_log_probs():
    from beamroot.growth import tree_split
    torch.manual_seed(0)
    tree = cifar_tree()
    x = torch.randn(5, 3, 32, 32)
    with torch.no_grad():
        before = tree.nodes["root"](x)
    new = tree_split(tree, "root", [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], "b2")
    assert tree.graph.children("root") == list(new) and len(new) == 2
    cache = tree.node_cache(x)
    with torch.no_grad():
        for leaf in new:
            assert torch.allclose(cache.output(leaf), before, atol=1e-6)
        route = cache.output("root")
    assert route.shape == (5, 2)
    assert torch.allclose(route.exp().sum(1), torch.ones(5), atol=1e-6)
    assert tree.split_blocks == {"root": "b2"}


GROUPS = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def split_tree(block="b2", width=16):
    from beamroot.growth import tree_split
    torch.manual_seed(0)
    tree = cifar_tree(width)
    tree_split(tree, "root", GROUPS, block)
    with torch.no_grad():  # make the two leaves differ, as training would
        for p in tree.nodes["leaf0"].parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return tree


def test_route_on_a_split_tree_gives_reach_route_and_the_soft_mixture():
    tree = split_tree()
    x = torch.randn(6, 3, 32, 32)
    with torch.no_grad():
        out = tree.route(x)
        cache = tree.node_cache(x)
        route = cache.output("root")
        leaves = {n: F.log_softmax(cache.output(n), 1) for n in ("leaf0", "leaf1")}
    assert torch.allclose(out["log_route"]["root"], route)
    assert torch.equal(out["log_reach"]["root"], torch.zeros(6))
    for i, n in enumerate(("leaf0", "leaf1")):
        assert torch.allclose(out["log_reach"][n], route[:, i])
        assert torch.allclose(out["leaf_log_probs"][n], leaves[n])
    mix = route[:, :1].exp() * leaves["leaf0"].exp() + route[:, 1:].exp() * leaves["leaf1"].exp()
    assert torch.allclose(out["log_probs"].exp(), mix, atol=1e-6)


def test_beam_one_follows_the_router_argmax():
    tree = split_tree()
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        cache = tree.node_cache(x)
        pick = cache.output("root").argmax(1)
        logits = torch.stack([cache.output("leaf0"), cache.output("leaf1")])
    expected = logits[pick, torch.arange(8)].argmax(1).tolist()
    results = beam_search(tree, x, 1)
    assert [r.pred for r in results] == expected
    assert [r.evaluated for r in results] == [["root", ["leaf0", "leaf1"][i]] for i in pick]


def test_beam_one_macs_are_trunk_plus_router_head_plus_one_branch():
    tree = split_tree("b2")
    img = torch.zeros(1, 3, 32, 32)
    router, branch = tree.nodes["root"], tree.nodes["leaf0"]
    with torch.no_grad():
        trunk_out = router.segment(img)
    parts = (module_macs(router.segment, img) + module_macs(router.head, trunk_out)
             + module_macs(branch, trunk_out))
    _, nodes = model_costs(tree)
    (res,) = beam_search(tree, torch.randn(1, 3, 32, 32), 1)
    assert sum(nodes[n] for n in res.evaluated) == parts
    assert module_macs(branch.segment, trunk_out) > 0  # the branch owns b3 and b4


def test_split_tree_checkpoint_round_trip_keeps_split_blocks_and_groups(tmp_path):
    tree = split_tree("b3")
    torch.save(tree.checkpoint(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "cifar10"}))
    back, _ = load_run(tmp_path, "cpu")
    assert back.split_blocks == {"root": "b3"} and back.groups == {"root": GROUPS}
    x = torch.randn(4, 3, 32, 32)
    assert torch.equal(tree.route(x)["log_probs"], back.route(x)["log_probs"])


def test_teacher_forced_loss_trains_only_the_nodes_on_each_images_true_path():
    tree = split_tree("b2").train()
    x, y = torch.randn(6, 3, 32, 32), torch.tensor([0, 1, 2, 3, 4, 0])  # all group 0
    tree.teacher_forced_loss(x, y).backward()
    assert all(p.grad is None or not p.grad.any() for p in tree.nodes["leaf1"].parameters())
    assert all(p.grad is not None and p.grad.any() for p in tree.nodes["leaf0"].head.parameters())
    assert all(p.grad is not None for p in tree.nodes["root"].parameters())


def test_teacher_forced_loss_is_router_ce_on_group_labels_plus_leaf_ce_per_image():
    tree = split_tree("b2")
    x, y = torch.randn(6, 3, 32, 32), torch.tensor([0, 7, 2, 9, 5, 1])
    with torch.no_grad():
        cache = tree.node_cache(x)
        g = torch.tensor([0, 1, 0, 1, 1, 0])
        router = F.nll_loss(cache.output("root"), g, reduction="sum")
        leaf = (F.cross_entropy(cache.output("leaf0")[g == 0], y[g == 0], reduction="sum")
                + F.cross_entropy(cache.output("leaf1")[g == 1], y[g == 1], reduction="sum"))
        loss = tree.teacher_forced_loss(x, y)
    assert torch.allclose(loss, (router + leaf) / 6, atol=1e-5)


def test_teacher_forced_training_learns_the_groups_at_b2():
    from beamroot.growth import tree_split
    torch.manual_seed(0)
    tree = TreeModel(width=4)
    tree_split(tree, "root", GROUPS, "b2")
    gen = torch.Generator().manual_seed(1)
    protos = torch.randn(10, 3, 32, 32, generator=gen)
    protos[:5, 0] += 1.0  # group 0 is redder

    def batch(n=64):
        y = torch.randint(0, 10, (n,), generator=gen)
        return protos[y] + 0.5 * torch.randn(n, 3, 32, 32, generator=gen), y

    opt = torch.optim.Adam(tree.parameters(), lr=3e-3)
    tree.train()
    for _ in range(30):
        x, y = batch()
        opt.zero_grad()
        tree.teacher_forced_loss(x, y).backward()
        opt.step()
    x, y = batch(256)
    with torch.no_grad():
        picked = tree.eval().route(x)["log_route"]["root"].argmax(1)
    assert (picked == (y >= 5).long()).float().mean() > 0.8


def test_a_child_leaf_splits_from_its_own_segment_and_round_trips(tmp_path):
    from beamroot.growth import tree_split
    tree = split_tree("b2")
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        before = tree.node_cache(x).output("leaf0")
    with pytest.raises(ValueError):
        tree_split(tree, "leaf0", [[0, 1], [2, 3, 4]], "b1")  # b1 is the parent's
    with pytest.raises(ValueError):
        tree_split(tree, "leaf0", [[0, 1], [2, 3, 5]], "b3")  # 5 never reaches leaf0
    a, b = tree_split(tree, "leaf0", [[0, 1], [2, 3, 4]], "b3")
    assert tree.node_classes(b) == [2, 3, 4] and tree.start_block(a) == 3
    with torch.no_grad():
        assert torch.allclose(tree.node_cache(x).output(a), before, atol=1e-6)
    # a batch of group-1 images never reaches the inner router: no empty-batch error
    tree.teacher_forced_loss(x, torch.tensor([5, 6, 7, 8])).backward()
    assert all(p.grad is None for p in tree.nodes["leaf0"].parameters())
    torch.save(tree.checkpoint(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "cifar10"}))
    back, _ = load_run(tmp_path, "cpu")
    assert back.split_blocks == {"root": "b2", "leaf0": "b3"}
    assert torch.equal(tree.eval()(x), back(x))


def test_train_run_trains_a_split_tree_logs_routing_and_reloads(tmp_path):
    from beamroot.growth import tree_split

    def split_after_first(model, epoch):
        if epoch == 1:
            tree_split(model, "root", GROUPS, "b2")
            return True
        return False

    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "2", "--limit-batches", "2", "--width", "4",
                        "--grow-every", "0",
                        "--runs-dir", str(tmp_path)]), after_epoch=split_after_first)["run_dir"]
    rows = [json.loads(s) for s in (d / "metrics.jsonl").read_text().splitlines()]
    routing = rows[-1]["routing"]
    assert 0 <= routing["root"]["entropy"] <= 1 and len(routing["root"]["usage"]) == 2
    assert all(0 < routing[n]["purity"] <= 1 for n in ("leaf0", "leaf1"))
    state = torch.load(d / "model.pt")["state_dict"]
    head = "nodes.{}.head.linear.weight"
    assert not torch.equal(state[head.format("leaf0")], state[head.format("leaf1")])  # trained
    model, _ = load_run(d, "cpu")
    assert model.split_blocks == {"root": "b2"}
    rows = evaluate_main([str(d), "--device", "cpu", "--beams", "1", "--limit-batches", "1"])
    assert rows[0]["mode"] == "beam 1"


def test_confusion_groups_cut_two_blocks_of_mutually_confused_classes():
    from beamroot.growth import confusion_groups
    classes = [1, 3, 4, 6, 8]  # groups {1, 4, 8} and {3, 6}, interleaved
    conf = torch.eye(5) * 50
    for a, b in [(0, 2), (0, 4), (2, 4), (1, 3)]:
        conf[a, b] = conf[b, a] = 10
    conf[0, 1] = 1  # a little cross-group confusion
    assert confusion_groups(conf, classes) == [[1, 4, 8], [3, 6]]


def test_confusion_groups_of_two_classes_are_two_singletons():
    from beamroot.growth import confusion_groups
    assert confusion_groups(torch.tensor([[9.0, 1.0], [2.0, 8.0]]), [7, 2]) == [[2], [7]]


def one_class_confused_with_all(classes):
    """Class `classes[0]` is a little, equally confused with every other class; the rest
    form blocks {1, 2, 3} and {4, 5} (positions) with some confusion between the blocks."""
    n = len(classes)
    conf = torch.eye(n) * 100
    for i in range(1, n):
        conf[0, i] = conf[i, 0] = 2
    for a, b in [(1, 2), (1, 3), (2, 3), (4, 5)]:
        if b < n:
            conf[a, b] = conf[b, a] = 20
    for a in (1, 2, 3):
        for b in (4, 5):
            if b < n:
                conf[a, b] = conf[b, a] = 6
    return conf


def test_confusion_groups_of_four_or_more_classes_have_no_singleton():
    from beamroot.growth import confusion_groups
    classes = [0, 2, 3, 5, 7, 9]
    # The top cut peels class 0 off; the best qualifying cut has mean dissimilarity 0.9675
    # against 0.9653 for the runner-up [[0, 2, 3, 5], [7, 9]].
    assert confusion_groups(one_class_confused_with_all(classes), classes) \
        == [[0, 7, 9], [2, 3, 5]]


def test_confusion_groups_of_three_classes_may_still_give_a_singleton():
    from beamroot.growth import confusion_groups
    conf = one_class_confused_with_all([4, 1, 8])
    assert confusion_groups(conf, [4, 1, 8]) == [[1, 8], [4]]


def test_choose_split_block_picks_the_earliest_block_at_or_above_the_bar():
    from beamroot.growth import choose_split_block
    accs = {"b2": 0.90, "b3": 0.93, "b4": 0.97}
    assert choose_split_block(accs, 0.93) == "b3"


def test_choose_split_block_has_no_fallback_when_no_block_passes():
    from beamroot.growth import choose_split_block
    assert choose_split_block({"b1": 0.7, "b2": 0.95, "b3": 0.8}, 0.99) is None


def group_data(n=512, seed=1):
    """Ten classes in GROUPS: the groups differ in redness (separable from pixels), classes
    inside a group barely differ, so a leaf confuses classes within a group only."""
    from torch.utils.data import DataLoader, TensorDataset
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(2, 3, 32, 32, generator=gen)
    base[0, 0] += 1.5
    protos = base[[0] * 5 + [1] * 5] + 0.3 * torch.randn(10, 3, 32, 32, generator=gen)
    y = torch.randint(0, 10, (n,), generator=gen)
    x = protos[y] + 0.5 * torch.randn(n, 3, 32, 32, generator=gen)
    train = DataLoader(TensorDataset(x[:384], y[:384]), batch_size=64, shuffle=True,
                       generator=torch.Generator().manual_seed(0))
    return train, DataLoader(TensorDataset(x[384:], y[384:]), batch_size=64)


def trained_leaf(train, steps=3):
    torch.manual_seed(0)
    tree = TreeModel(width=4)
    opt = torch.optim.Adam(tree.parameters(), lr=3e-3)
    for _ in range(steps):
        for x, y in train:
            opt.zero_grad()
            tree.teacher_forced_loss(x, y).backward()
            opt.step()
    return tree


def route_acc(tree, router, loader):
    hits = total = 0
    with torch.no_grad():
        for x, y in loader:
            pick = tree.eval().node_cache(x).output(router).argmax(1)
            hits += (pick == (y >= 5).long()).sum().item()
            total += len(y)
    return hits / total


def test_grow_splits_a_tree_leaf_on_its_confusion_groups_at_a_well_routed_block(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = trained_leaf(train)
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=6,
                 val_loader=val, route_acc_min=0.9, plateau_patience=0)
    assert event["split"] == "root" and event["groups"] == GROUPS
    assert list(event["block_val_acc"]) == ["b1", "b2", "b3", "b4"]
    assert "fallback" not in event and event["excluded_blocks"] == ["b4"]
    assert event["block_val_acc"][event["block"]] >= 0.9
    assert all(event["block_val_acc"][b] < 0.9 for b in event["block_val_acc"]
               if b < event["block"])
    assert tree.split_blocks == {"root": event["block"]} and tree.groups == {"root": GROUPS}
    assert route_acc(tree, "root", val) >= 0.9  # the router starts from the probe head
    logged = json.loads((tmp_path / "growth.jsonl").read_text().splitlines()[-1])
    assert {"groups", "block", "block_val_acc", "plateau_losses"} <= set(logged)
    assert (tmp_path / "graphs" / "epoch_001.json").exists()


def test_tree_growth_leaves_training_rng_untouched(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = trained_leaf(train)
    torch.manual_seed(123)
    cpu, shuffle = torch.get_rng_state(), train.generator.get_state()
    assert grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=6,
                val_loader=val, plateau_patience=0) is not None
    assert torch.equal(torch.get_rng_state(), cpu)
    assert torch.equal(train.generator.get_state(), shuffle)


def test_grow_never_chooses_a_one_class_leaf(tmp_path):
    from beamroot.growth import grow, leaf_losses, tree_split
    train, val = group_data()
    tree = trained_leaf(train, steps=1)
    tree_split(tree, "root", [list(range(9)), [9]], "b2")
    with torch.no_grad():  # the one-class leaf now has by far the highest loss
        tree.nodes["leaf1"].head.linear.weight.mul_(100)
    losses = leaf_losses(tree, train, "cpu", 6)
    assert max(losses, key=losses.get) == "leaf1"
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6, val_loader=val,
                 plateau_patience=0)
    assert event["split"] == "leaf0" and set(event["block_val_acc"]) == {"b3", "b4"}
    assert grow(tree, train, "cpu", 2, tmp_path, max_nodes=7, limit_batches=6,
                val_loader=val, plateau_patience=0) is not None  # leaf1 is still never picked
    assert tree.graph.nodes["leaf1"].kind == "leaf"


def test_train_run_grows_a_tree_and_logs_each_split(tmp_path, capsys):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "4", "--width", "4", "--max-nodes", "5",
                        "--route-acc-min", "0.5", "--plateau-patience", "0",
                        "--post-split-epochs", "2", "--runs-dir", str(tmp_path)]))["run_dir"]
    logged = [json.loads(s) for s in (d / "growth.jsonl").read_text().splitlines()]
    assert [(e["epoch"], e.get("skipped")) for e in logged] == [(1, None), (2, "below_bar"),
                                                                 (3, None)]
    events = [e for e in logged if "split" in e]
    for e in events:
        assert {"groups", "block", "block_val_acc"} <= set(e) and "fallback" not in e
        assert e["block"] in e["block_val_acc"]
    assert sorted(c for g in events[0]["groups"] for c in g) == list(range(10))
    assert (d / "graphs" / "epoch_003.json").exists()
    assert not (d / "graphs" / "epoch_002.json").exists()  # a skip changes nothing
    model, _ = load_run(d, "cpu")
    assert len(model.graph.nodes) == 5 and model.split_blocks[events[0]["split"]] == events[0]["block"]
    out = capsys.readouterr().out
    assert f"split root -> {events[0]['new_nodes']} at {events[0]['block']}" in out


def test_route_acc_min_exits_for_other_models(tmp_path):
    with pytest.raises(SystemExit, match="tree"):
        run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--epochs", "1",
                        "--route-acc-min", "0.5", "--runs-dir", str(tmp_path)]))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("size", [7, 8])
def test_router_head_pool_matches_adaptive_pool_and_runs_on_mps(size):
    from beamroot.models.tree import RouterHead
    head = RouterHead(3, size, 2)
    x = torch.randn(2, 3, size, size)
    ref = F.adaptive_avg_pool2d(x, 4)
    assert torch.allclose(head.pool(x), ref, atol=1e-6)
    if size % 4 == 0:
        assert torch.equal(head.pool(x), ref)
    if torch.backends.mps.is_available():
        out = head.to("mps")(x.to("mps")).cpu()
        assert torch.allclose(out, head.cpu()(x), atol=1e-5)


def test_a_root_below_the_bar_is_not_split_until_a_later_step_passes(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = trained_leaf(train)
    torch.manual_seed(123)
    cpu, shuffle = torch.get_rng_state(), train.generator.get_state()
    skip = grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=6,
                val_loader=val, route_acc_min=1.01, plateau_patience=0)  # no block can reach it
    assert torch.equal(torch.get_rng_state(), cpu)  # the skip path leaves the RNG too
    assert torch.equal(train.generator.get_state(), shuffle)
    assert skip["skipped"] == "below_bar" and "split" not in skip
    assert skip["leaf"] == "root" and skip["groups"] == GROUPS
    assert list(skip["block_val_acc"]) == ["b1", "b2", "b3", "b4"]
    assert list(tree.graph.nodes) == ["root"] and tree.split_blocks == {}
    assert not (tmp_path / "graphs").exists()
    event = grow(tree, train, "cpu", 2, tmp_path, max_nodes=3, limit_batches=6,
                 val_loader=val, route_acc_min=0.9, plateau_patience=0)
    assert event["split"] == "root" and "fallback" not in event
    logged = [json.loads(s) for s in (tmp_path / "growth.jsonl").read_text().splitlines()]
    assert [e.get("skipped") for e in logged] == ["below_bar", None]
    assert logged[0]["epoch"] == 1 and logged[0]["block_val_acc"] == skip["block_val_acc"]


def test_a_non_root_leaf_below_the_bar_is_skipped_not_split(tmp_path):
    from beamroot.growth import grow, tree_split
    train, val = group_data()
    tree = trained_leaf(train, steps=1)
    tree_split(tree, "root", GROUPS, "b2")
    torch.manual_seed(123)
    cpu, shuffle = torch.get_rng_state(), train.generator.get_state()
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6,
                 val_loader=val, route_acc_min=1.01, plateau_patience=0)
    assert torch.equal(torch.get_rng_state(), cpu)
    assert torch.equal(train.generator.get_state(), shuffle)
    assert event["leaf"] in ("leaf0", "leaf1") and event["skipped"] == "below_bar"
    assert "split" not in event and "fallback" not in event and "excluded_blocks" not in event
    assert list(event["block_val_acc"]) == ["b3", "b4"]
    assert list(tree.graph.nodes) == ["root", "leaf0", "leaf1"]
    logged = json.loads((tmp_path / "growth.jsonl").read_text())
    assert logged["skipped"] == "below_bar" and "fallback" not in logged


def test_train_run_keeps_the_plain_network_while_the_root_misses_the_bar(tmp_path, capsys):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "3", "--width", "4", "--limit-batches", "2",
                        "--route-acc-min", "1.01", "--plateau-patience", "0",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    events = [json.loads(s) for s in (d / "growth.jsonl").read_text().splitlines()]
    assert [(e["epoch"], e["skipped"]) for e in events] == [(1, "below_bar"),
                                                            (2, "below_bar")]
    model, _ = load_run(d, "cpu")
    assert list(model.nodes) == ["root"]
    assert "root below route acc 1.01" in capsys.readouterr().out


def test_choose_root_block_never_picks_the_last_block():
    from beamroot.growth import choose_root_block
    accs = {"b1": 0.90, "b2": 0.91, "b3": 0.92, "b4": 0.95}
    assert choose_root_block(accs, 0.93) is None  # only b4 passes: skip
    assert choose_root_block({**accs, "b2": 0.94}, 0.93) == "b2"


def test_keep_channels_keeps_the_fewest_channels_carrying_the_share():
    from beamroot.growth import keep_channels
    mass = torch.tensor([5.0, 0.0, 3.0, 1.0, 0.0, 1.0])
    assert keep_channels(mass, 0.95).tolist() == [0, 2, 3, 5]
    assert keep_channels(mass, 0.5).tolist() == [0]
    assert keep_channels(mass, 0.8).tolist() == [0, 2]
    assert keep_channels(mass, 1.0).tolist() == [0, 1, 2, 3, 4, 5]  # full copy
    assert keep_channels(torch.zeros(4), 0.95).tolist() == [0]  # never drop every channel


def silence(conv, channels):
    """Make output `channels` of `conv` zero after its ReLU, for every input."""
    with torch.no_grad():
        conv.weight[channels] = 0
        conv.bias[channels] = -1


def narrowed_tree(x):
    """Width-4 tree (4/8/16/32/32 channels per conv) split at b2; leaf0 drops channels that
    are zero for every input, leaf1 keeps all. Returns the tree and the root's output on x
    before the split."""
    from beamroot.growth import tree_split
    torch.manual_seed(0)
    tree = cifar_tree(width=4)
    b3, b4 = tree.nodes["root"].segment[2], tree.nodes["root"].segment[3]
    silence(b3[0], [1, 5])
    silence(b4[0], [0, 3, 7])
    silence(b4[2], [2])
    with torch.no_grad():
        before = tree.nodes["root"](x)

    def drop(n, gone):
        return torch.tensor([c for c in range(n) if c not in gone])

    keep = [[drop(16, [1, 5]), drop(32, [0, 3, 7]), drop(32, [2])],
            [torch.arange(16), torch.arange(32), torch.arange(32)]]
    tree_split(tree, "root", GROUPS, "b2", keep=keep)
    return tree, before


def test_a_narrowed_branch_equals_the_full_branch_when_dropped_channels_are_zero():
    x = torch.randn(5, 3, 32, 32)
    tree, before = narrowed_tree(x)
    narrow, full = "leaf0", "leaf1"
    convs = [m for m in tree.nodes[narrow].segment.modules() if isinstance(m, nn.Conv2d)]
    assert [(c.in_channels, c.out_channels) for c in convs] == [(8, 14), (14, 29), (29, 31)]
    assert tree.nodes[narrow].head.linear.in_features == 31
    with torch.no_grad():
        cache = tree.node_cache(x)
        assert torch.allclose(cache.output(narrow), before, atol=1e-5)
        assert torch.allclose(cache.output(full), before, atol=1e-6)


def test_narrowed_branches_round_trip_through_load_run(tmp_path):
    from beamroot.growth import tree_split
    x = torch.randn(4, 3, 32, 32)
    tree, _ = narrowed_tree(x)
    tree_split(tree, "leaf0", [[0, 1], [2, 3, 4]], "b3",  # the router keeps leaf0's 14 b3 channels
               keep=[[torch.arange(10), torch.arange(5)], [torch.arange(29), torch.arange(31)]])
    torch.save(tree.checkpoint(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "cifar10"}))
    back, _ = load_run(tmp_path, "cpu")
    assert {k: v.shape for k, v in back.state_dict().items()} == \
        {k: v.shape for k, v in tree.state_dict().items()}
    assert torch.equal(tree.eval()(x), back(x))


def test_a_checkpoint_without_channel_widths_still_loads(tmp_path):
    tree = split_tree("b3", width=4)
    ckpt = tree.checkpoint()
    ckpt.pop("channels", None)  # written before branch keep existed
    torch.save(ckpt, tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "cifar10"}))
    back, _ = load_run(tmp_path, "cpu")
    x = torch.randn(4, 3, 32, 32)
    assert torch.equal(tree(x), back(x))


def test_beam_one_macs_of_each_path_use_that_branchs_width():
    x = torch.randn(1, 3, 32, 32)
    tree, _ = narrowed_tree(x)
    _, nodes = model_costs(tree)
    img = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        trunk_out = tree.nodes["root"].segment(img)
    for leaf in ("leaf0", "leaf1"):
        assert nodes[leaf] == module_macs(tree.nodes[leaf], trunk_out)
    assert nodes["leaf0"] < nodes["leaf1"]  # leaf1 is a full copy, leaf0 narrowed
    full = cifar_tree(width=4)
    assert nodes["leaf1"] + nodes["root"] > model_costs(full)[1]["root"]  # plus router head
    with torch.no_grad():  # route every image to leaf0
        tree.nodes["root"].head.net[-1].bias.copy_(torch.tensor([100.0, -100.0]))
    (res,) = beam_search(tree, x, 1)
    assert res.evaluated == ["root", "leaf0"]


def test_grow_narrows_each_branch_to_its_groups_activation_share_and_logs_it(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = trained_leaf(train)
    silence(tree.nodes["root"].segment[3][2], list(range(16)))  # half of b4's last conv
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=6,
                 val_loader=val, route_acc_min=0.9, branch_keep=0.95, plateau_patience=0)
    new = event["new_nodes"]
    assert event["branch_keep"] == 0.95
    assert set(event["branch_channels"]) == set(new)
    for leaf in new:
        kept, full = event["branch_channels"][leaf], tree.default_widths(leaf)
        assert kept == tree.node_widths(leaf) and len(kept) == len(full)
        assert all(1 <= k <= f for k, f in zip(kept, full))
        assert kept[-1] <= 16  # a channel no image activates is always dropped
    _, nodes = model_costs(tree)
    assert event["branch_macs"] == {leaf: nodes[leaf] for leaf in new}
    logged = json.loads((tmp_path / "growth.jsonl").read_text().splitlines()[-1])
    assert logged["branch_channels"] == event["branch_channels"]
    x, _ = next(iter(val))
    assert tree.eval()(x).shape == (len(x), 10)


def test_grow_with_branch_keep_one_copies_the_branches_at_full_width(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = trained_leaf(train)
    x, _ = next(iter(val))
    with torch.no_grad():
        before = tree.eval().nodes["root"](x)
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=6,
                 val_loader=val, route_acc_min=0.9, branch_keep=1.0, plateau_patience=0)
    assert tree.channels == {}
    with torch.no_grad():
        for leaf in event["new_nodes"]:
            assert event["branch_channels"][leaf] == tree.default_widths(leaf)
            assert torch.equal(tree.node_cache(x).output(leaf), before)


def test_branch_keep_exits_for_other_models(tmp_path):
    with pytest.raises(SystemExit, match="tree"):
        run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--epochs", "1",
                        "--branch-keep", "0.9", "--runs-dir", str(tmp_path)]))
    assert not list(tmp_path.iterdir())


def test_train_run_passes_branch_keep_to_tree_growth(tmp_path, capsys):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "2", "--width", "4", "--max-nodes", "3",
                        "--route-acc-min", "0.5", "--plateau-patience", "0", "--branch-keep", "0.5",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    (event,) = [json.loads(s) for s in (d / "growth.jsonl").read_text().splitlines()]
    assert event["branch_keep"] == 0.5
    model, _ = load_run(d, "cpu")
    assert {n: model.node_widths(n) for n in event["new_nodes"]} == event["branch_channels"]
    assert "channels" in capsys.readouterr().out


def test_probes_train_on_all_true_path_training_images_by_default(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()  # 384 training images in 6 batches of 64
    tree = trained_leaf(train)
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=1,
                 val_loader=val, route_acc_min=0.5, plateau_patience=0)
    assert event["probe_images"] == 384  # not the one growth batch the leaf losses use
    tree = trained_leaf(train)
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=1,
                 val_loader=val, route_acc_min=0.5, probe_batches=2, plateau_patience=0)
    assert event["probe_images"] == 128


@pytest.mark.parametrize("flags", [["--probe-batches", "3"], ["--post-split-epochs", "2"]])
def test_fair_growth_flags_exit_for_other_models(tmp_path, flags):
    with pytest.raises(SystemExit, match="tree"):
        run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--epochs", "1",
                        "--runs-dir", str(tmp_path), *flags]))
    assert not list(tmp_path.iterdir())


def test_a_tree_run_trains_post_split_epochs_past_each_split(tmp_path):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "2", "--width", "4", "--max-nodes", "7",
                        "--route-acc-min", "0.5", "--plateau-patience", "0",
                        "--post-split-epochs", "3", "--runs-dir", str(tmp_path)]))["run_dir"]
    logged = [json.loads(s) for s in (d / "growth.jsonl").read_text().splitlines()]
    splits = [e["epoch"] for e in logged if "split" in e]
    assert splits == [1, 3, 4]  # epoch 2 skipped; the later splits come in extended epochs
    rows = [json.loads(s) for s in (d / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2, 3, 4, 5, 6, 7]  # last split 4 + 3
    assert rows[-1]["epochs_trained"] == 7


def test_a_tree_run_that_never_splits_ends_at_epochs(tmp_path):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "2", "--limit-batches", "1", "--max-nodes", "1",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    rows = [json.loads(s) for s in (d / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2] and rows[-1]["epochs_trained"] == 2


@pytest.mark.parametrize("flags, images", [([], 224), (["--probe-batches", "1"], 128)])
def test_train_run_passes_probe_batches_to_tree_growth(tmp_path, flags, images):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "2", "--width", "4", "--max-nodes", "3",
                        "--growth-batches", "1", "--route-acc-min", "0.5",
                        "--plateau-patience", "0",
                        "--runs-dir", str(tmp_path), *flags]))["run_dir"]
    (event,) = [json.loads(s) for s in (d / "growth.jsonl").read_text().splitlines()]
    assert event["probe_images"] == images  # 224 training images in batches of 128


def test_a_tree_defaults_to_width_16_and_other_models_keep_the_dataset_width(tmp_path):
    from beamroot.train import build_model
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "1", "--limit-batches", "1", "--max-nodes", "1",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    assert json.loads((d / "config.json").read_text())["width"] == 16
    assert load_run(d, "cpu")[0].config["width"] == 16
    assert build_model("tree", "cifar10").config["width"] == 16
    assert build_model("tree", "cifar10", width=8).config["width"] == 8
    assert build_model("dag", "cifar10").config["width"] == 32


def test_a_leaf_plateaus_once_its_loss_improves_less_than_delta_over_patience_epochs():
    from beamroot.growth import plateaued
    history = [1.0]
    for gain in (0.05, 0.03, 0.005, 0.004):
        history.append(history[-1] * (1 - gain))
    assert [plateaued(history[:n], 0.01, 2) for n in range(1, 6)] == [False] * 4 + [True]
    assert not plateaued([1.0, 1.0], 0.01, 2)  # fewer than patience + 1 measurements
    assert not plateaued([], 0.01, 2)
    assert plateaued([1.0, 1.2, 1.1], 0.01, 2)  # no net improvement
    assert plateaued([0.7], 0.01, 0)  # patience 0: a leaf acts at its first measurement


def test_the_first_plateaued_leaf_acts_even_when_another_has_the_higher_loss(tmp_path):
    from beamroot.growth import grow, leaf_losses, tree_split
    train, val = group_data()
    tree = trained_leaf(train, steps=1)
    tree_split(tree, "root", GROUPS, "b2")
    losses = leaf_losses(tree, train, "cpu", 6)
    worse, calm = sorted(losses, key=losses.get, reverse=True)
    history = {worse: [1e3], calm: [1e-4]}  # the next measurement: worse improves, calm not
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6,
                 val_loader=val, route_acc_min=0.0, leaf_history=history, plateau_patience=1)
    assert event["split"] == calm and event["plateau_losses"][0] == 1e-4
    assert calm not in history and len(history[worse]) == 2
    assert not set(event["new_nodes"]) & set(history)  # new leaves start with no history


def test_no_plateaued_leaf_waits_without_logging_or_touching_the_rng(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = trained_leaf(train)
    history = {}
    torch.manual_seed(123)
    cpu, shuffle = torch.get_rng_state(), train.generator.get_state()
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=6,
                 val_loader=val, route_acc_min=0.0, leaf_history=history)
    assert torch.equal(torch.get_rng_state(), cpu)
    assert torch.equal(train.generator.get_state(), shuffle)
    assert event["waiting"] == ["root"] and set(event["val_losses"]) == {"root"}
    assert len(history["root"]) == 1 and list(tree.graph.nodes) == ["root"]
    assert not (tmp_path / "growth.jsonl").exists()


def test_a_tree_run_waits_for_a_leaf_to_settle_and_skips_do_not_move_the_end(tmp_path, capsys):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "5", "--width", "4", "--limit-batches", "2",
                        "--max-nodes", "3", "--plateau-delta", "1.0",
                        "--route-acc-min", "1.01", "--runs-dir", str(tmp_path)]))["run_dir"]
    events = [json.loads(s) for s in (d / "growth.jsonl").read_text().splitlines()]
    # default patience 2: the root first acts at its third measurement, epoch 3
    assert [(e["epoch"], e["skipped"]) for e in events] == [(3, "below_bar"), (4, "below_bar")]
    assert len(events[0]["plateau_losses"]) == 3
    rows = [json.loads(s) for s in (d / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["epochs_trained"] == 5  # skips never extend the run
    out = capsys.readouterr().out
    assert "waiting: root" in out and "root below route acc 1.01" in out
    config = json.loads((d / "config.json").read_text())
    assert config["tree_growth"] == "settle" and config["plateau_patience"] == 2


@pytest.mark.parametrize("flags", [["--tree-growth", "settle"], ["--tree-growth", "progressive"],
                                   ["--plateau-delta", "0.05"],
                                   ["--plateau-patience", "3"]])
def test_settle_flags_exit_for_other_models(tmp_path, flags):
    with pytest.raises(SystemExit, match="tree"):
        run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--epochs", "1",
                        "--runs-dir", str(tmp_path), *flags]))
    assert not list(tmp_path.iterdir())


def progressive_tree(width=16):
    from beamroot.train import build_model
    return build_model("tree", "cifar10", width=width, tree_growth="progressive").eval()


def test_a_progressive_root_starts_as_b1_plus_the_classifier(tmp_path):
    tree = progressive_tree()
    root = tree.nodes["root"]
    assert len(root.segment) == 1 and tree.end_block("root") == 1
    b1_and_head = nn.Sequential(*deep4_blocks(3, 16)[:1], nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(16, 10))
    assert model_costs(tree)[1]["root"] == module_macs(b1_and_head, torch.zeros(1, 3, 32, 32))
    assert tree(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert tree.block_width("root", 0) == 16 and root.head.linear.in_features == 16
    torch.save(tree.checkpoint(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "cifar10"}))
    back, _ = load_run(tmp_path, "cpu")
    assert back.leaf_ends == {"root": 1}
    x = torch.randn(3, 3, 32, 32)
    assert torch.equal(tree(x), back(x))


def test_deepen_narrows_the_newest_block_and_adds_the_next_one_reading_the_kept_channels(tmp_path):
    from beamroot.growth import tree_deepen
    torch.manual_seed(0)
    tree = progressive_tree(width=4)
    silence(tree.nodes["root"].segment[0][0], [1, 3])  # b1's one conv: channels 1 and 3 are dead
    x = torch.randn(5, 3, 32, 32)
    with torch.no_grad():
        full_b1 = tree.nodes["root"].segment[0](x)
    old_conv = tree.nodes["root"].segment[0][0]
    kept = torch.tensor([0, 2])
    assert tree_deepen(tree, "root", keep=[kept]) == "b2"
    node = tree.nodes["root"]
    assert tree.end_block("root") == 2 and tree.leaf_ends == {"root": 2}
    assert len(node.segment) == 2 and tree.node_widths("root") == [2, 8]
    assert tree.channels == {"root": [2, 8]}
    with torch.no_grad():  # the narrowed block's output is the full block's on its kept channels
        assert torch.equal(node.segment[0](x), full_b1[:, kept])
    assert torch.equal(node.segment[0][0].weight, old_conv.weight[kept])  # weights kept exactly
    assert torch.equal(node.segment[0][0].bias, old_conv.bias[kept])
    assert node.segment[1][0].in_channels == 2 and node.segment[1][0].out_channels == 8
    assert node.head.linear.in_features == 8  # a fresh classifier on the new block's width
    assert tree.block_width("root", 1) == 8 and tree(x).shape == (5, 10)
    torch.save(tree.checkpoint(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "cifar10"}))
    back, _ = load_run(tmp_path, "cpu")
    assert back.leaf_ends == {"root": 2} and back.channels == {"root": [2, 8]}
    assert torch.equal(tree.eval()(x), back(x))
    with pytest.raises(ValueError):  # b4 is the last block
        for _ in range(4):
            tree_deepen(tree, "root")


def test_a_split_inside_a_shallow_leaf_gives_children_ending_at_its_last_block(tmp_path):
    from beamroot.growth import tree_deepen, tree_split
    torch.manual_seed(0)
    tree = progressive_tree(width=4)
    tree_deepen(tree, "root")
    tree_deepen(tree, "root")  # root owns b1..b3
    with pytest.raises(ValueError):
        tree_split(tree, "root", GROUPS, "b4")  # beyond the leaf's last block
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        before = tree.node_cache(x).output("root")
    a, b = tree_split(tree, "root", GROUPS, "b1")
    assert tree.leaf_ends == {a: 3, b: 3} and "root" not in tree.leaf_ends
    assert len(tree.nodes[a].segment) == 2 and tree.start_block(a) == 1
    with torch.no_grad():
        assert torch.allclose(tree.node_cache(x).output(a), before, atol=1e-6)
    c, d = tree_split(tree, a, [[0, 1], [2, 3, 4]], "b2")  # leaf a splits at its own b2
    assert tree.leaf_ends == {b: 3, c: 3, d: 3} and len(tree.nodes[c].segment) == 1
    (e, f) = tree_split(tree, c, [[0], [1]], "b3")  # splits at its last block: no blocks left
    assert len(tree.nodes[e].segment) == 0 and tree.start_block(e) == 3
    with torch.no_grad():
        assert tree.node_cache(x).output(e).shape == (4, 10)
    torch.save(tree.checkpoint(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps({"dataset": "cifar10"}))
    back, _ = load_run(tmp_path, "cpu")
    assert back.leaf_ends == tree.leaf_ends and torch.equal(tree.eval()(x), back(x))
    # a child of a split before b4 can deepen, whether it owns blocks or none
    assert tree_deepen(tree, e) == "b4" and tree_deepen(tree, b) == "b4"
    assert tree.end_block(e) == 4 and tree.leaf_ends == {d: 3, f: 3}


def shallow_leaf(train, ends=1, steps=2):
    """A width-4 progressive root trained a little, owning blocks up to index `ends`."""
    torch.manual_seed(0)
    tree = TreeModel(width=4, leaf_ends={"root": ends} if ends < 4 else None)
    opt = torch.optim.Adam(tree.parameters(), lr=3e-3)
    for _ in range(steps):
        for x, y in train:
            opt.zero_grad()
            tree.teacher_forced_loss(x, y).backward()
            opt.step()
    return tree


def test_a_plateaued_leaf_that_no_block_routes_deepens_and_logs_it(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = shallow_leaf(train)
    history = {}
    torch.manual_seed(123)
    cpu, shuffle = torch.get_rng_state(), train.generator.get_state()
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6, val_loader=val,
                 route_acc_min=1.01, plateau_patience=0, leaf_history=history, progressive=True)
    assert torch.equal(torch.get_rng_state(), cpu)  # deepening, init included, is seeded inside
    assert torch.equal(train.generator.get_state(), shuffle)
    assert event["leaf"] == "root" and event["deepen"] == "b2" and "split" not in event
    assert list(event["block_val_acc"]) == ["b1"] and event["epoch"] == 1
    assert tree.end_block("root") == 2 and list(tree.graph.nodes) == ["root"]
    assert event["kept_channels"] == tree.node_widths("root")[:1]  # the narrowed block b1
    assert event["node_macs"] == model_costs(tree)[1]["root"]
    assert "root" not in history  # the plateau history restarts after a deepen
    logged = json.loads((tmp_path / "growth.jsonl").read_text().splitlines()[-1])
    assert logged["deepen"] == "b2" and logged["kept_channels"] == event["kept_channels"]
    assert tree(next(iter(val))[0]).shape == (64, 10)


def test_deepen_keeps_the_channels_carrying_the_leafs_activation_share(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = shallow_leaf(train)
    silence(tree.nodes["root"].segment[0][0], [0, 1])  # b1's conv: two channels never fire
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6, val_loader=val,
                 route_acc_min=1.01, plateau_patience=0, progressive=True, branch_keep=0.95)
    assert event["kept_channels"][0] <= 2
    full = grow(shallow_leaf(train), train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6,
                val_loader=val, route_acc_min=1.01, plateau_patience=0, progressive=True,
                branch_keep=1.0)
    assert full["kept_channels"] == [4]  # branch keep 1.0 keeps the block whole


def test_a_plateaued_leaf_at_b4_that_no_block_routes_is_skipped_not_deepened(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = shallow_leaf(train, ends=4)
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6, val_loader=val,
                 route_acc_min=1.01, plateau_patience=0, progressive=True)
    assert event["skipped"] == "below_bar" and "deepen" not in event
    assert tree.end_block("root") == 4 and tree.leaf_ends == {}


def test_a_progressive_leaf_splits_at_the_earliest_block_that_passes(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = shallow_leaf(train, ends=3)
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6, val_loader=val,
                 route_acc_min=0.0, plateau_patience=0, progressive=True)
    assert event["split"] == "root" and event["block"] == "b1"
    assert list(event["block_val_acc"]) == ["b1", "b2", "b3"]
    assert event["excluded_blocks"] == ["b4"]
    assert tree.leaf_ends == {n: 3 for n in event["new_nodes"]}
    assert [len(tree.nodes[n].segment) for n in event["new_nodes"]] == [2, 2]  # b2, b3 copied


def test_the_progressive_root_may_split_at_its_last_block_unless_that_is_b4(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = shallow_leaf(train, ends=1)
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6, val_loader=val,
                 route_acc_min=0.0, plateau_patience=0, progressive=True)
    assert event["block"] == "b1" and tree.leaf_ends == {n: 1 for n in event["new_nodes"]}
    assert all(len(tree.nodes[n].segment) == 0 for n in event["new_nodes"])
    deep_root = shallow_leaf(train, ends=4)
    event = grow(deep_root, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6,
                 val_loader=val, route_acc_min=0.0, plateau_patience=0, progressive=True)
    assert event["block"] != "b4"


def test_a_child_with_no_blocks_deepens_at_its_plateau_and_a_leaf_at_b4_is_never_chosen(tmp_path):
    from beamroot.growth import grow, tree_split
    train, val = group_data()
    tree = shallow_leaf(train, ends=2)
    tree_split(tree, "root", GROUPS, "b2")  # both children own no blocks, last block b2
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=6, val_loader=val,
                 plateau_patience=0, progressive=True)  # room for no split: deepen only
    assert event["deepen"] == "b3" and event["kept_channels"] == []
    assert tree.end_block(event["leaf"]) == 3
    assert event["block_val_acc"] == {}  # nothing to probe: no block, no split possible
    other = ({"leaf0", "leaf1"} - {event["leaf"]}).pop()
    assert tree.start_block(other) == 2 and tree.end_block(other) == 2
    full = shallow_leaf(train, ends=4)
    tree_split(full, "root", GROUPS, "b4")  # children own nothing and end at b4: ineligible
    assert grow(full, train, "cpu", 2, tmp_path, max_nodes=9, limit_batches=6, val_loader=val,
                plateau_patience=0, progressive=True) is None


def test_progressive_growth_deepens_a_leaf_even_when_the_node_budget_is_spent(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = shallow_leaf(train, ends=1)
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=1, limit_batches=6, val_loader=val,
                 plateau_patience=0, progressive=True)
    assert event["deepen"] == "b2" and event["block_val_acc"] == {}
    assert grow(tree, train, "cpu", 2, tmp_path, max_nodes=1, limit_batches=6, val_loader=val,
                plateau_patience=0) is None  # settle growth with no room still does nothing


def test_a_progressive_run_deepens_to_b4_then_skips_and_each_deepen_moves_the_end(tmp_path, capsys):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--tree-growth", "progressive", "--epochs", "2", "--width", "4",
                        "--limit-batches", "2", "--max-nodes", "3", "--plateau-patience", "0",
                        "--route-acc-min", "1.01", "--post-split-epochs", "2",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    events = [json.loads(s) for s in (d / "growth.jsonl").read_text().splitlines()]
    assert [(e["epoch"], e.get("deepen"), e.get("skipped")) for e in events] == [
        (1, "b2", None), (2, "b3", None), (3, "b4", None), (4, None, "below_bar")]
    rows = [json.loads(s) for s in (d / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["epoch"] == rows[-1]["epochs_trained"] == 5  # the last deepen was epoch 3
    model, _ = load_run(d, "cpu")
    assert model.end_block("root") == 4 and model.leaf_ends == {}
    assert "deepen root with b2 narrowed b1" in capsys.readouterr().out
    assert json.loads((d / "config.json").read_text())["tree_growth"] == "progressive"


def test_a_progressive_run_splits_and_the_children_grow_their_own_blocks(tmp_path):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--tree-growth", "progressive", "--epochs", "6", "--width", "4",
                        "--limit-batches", "2", "--max-nodes", "3", "--plateau-patience", "0",
                        "--route-acc-min", "0.0", "--post-split-epochs", "2",
                        "--runs-dir", str(tmp_path)]))["run_dir"]
    events = [json.loads(s) for s in (d / "growth.jsonl").read_text().splitlines()]
    assert [(e["epoch"], e["block"]) for e in events if "split" in e] == [(1, "b1")]
    deepened = [e for e in events if "deepen" in e]  # no room to split: leaves only deepen
    assert [(e["leaf"], e["deepen"]) for e in deepened] == [("leaf0", "b2"), ("leaf0", "b3"),
                                                            ("leaf0", "b4")]
    model, _ = load_run(d, "cpu")  # (the synthetic val set holds no image of leaf1's classes)
    assert model.end_block("leaf0") == 4 and model.end_block("leaf1") == 1
    rows = [json.loads(s) for s in (d / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["epochs_trained"] == max(6, max(e["epoch"] for e in events) + 2)


def test_a_progressive_tree_that_can_never_split_is_the_plain_network(tmp_path):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--tree-growth", "progressive", "--epochs", "1", "--limit-batches", "1",
                        "--max-nodes", "1", "--runs-dir", str(tmp_path)]))["run_dir"]
    model, _ = load_run(d, "cpu")
    assert model.leaf_ends == {} and not (d / "growth.jsonl").exists()
    assert model_costs(model)[1] == model_costs(cifar_tree())[1]
