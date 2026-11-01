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


def test_choose_split_block_picks_the_earliest_block_at_or_above_the_bar():
    from beamroot.growth import choose_split_block
    accs = {"b2": 0.90, "b3": 0.93, "b4": 0.97}
    assert choose_split_block(accs, 0.93) == ("b3", None)


def test_choose_split_block_falls_back_to_the_last_block():
    from beamroot.growth import choose_split_block
    assert choose_split_block({"b1": 0.7, "b2": 0.95, "b3": 0.8}, 0.99) == ("b3", "last_block")


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
                 val_loader=val, route_acc_min=0.9)
    assert event["split"] == "root" and event["groups"] == GROUPS
    assert list(event["block_val_acc"]) == ["b1", "b2", "b3", "b4"]
    assert event["fallback"] is None
    assert event["block_val_acc"][event["block"]] >= 0.9
    assert all(event["block_val_acc"][b] < 0.9 for b in event["block_val_acc"]
               if b < event["block"])
    assert tree.split_blocks == {"root": event["block"]} and tree.groups == {"root": GROUPS}
    assert route_acc(tree, "root", val) >= 0.9  # the router starts from the probe head
    logged = json.loads((tmp_path / "growth.jsonl").read_text().splitlines()[-1])
    assert {"groups", "block", "block_val_acc", "fallback"} <= set(logged)
    assert (tmp_path / "graphs" / "epoch_001.json").exists()


def test_tree_growth_leaves_training_rng_untouched(tmp_path):
    from beamroot.growth import grow
    train, val = group_data()
    tree = trained_leaf(train)
    torch.manual_seed(123)
    cpu, shuffle = torch.get_rng_state(), train.generator.get_state()
    assert grow(tree, train, "cpu", 1, tmp_path, max_nodes=3, limit_batches=6,
                val_loader=val) is not None
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
    event = grow(tree, train, "cpu", 1, tmp_path, max_nodes=5, limit_batches=6, val_loader=val)
    assert event["split"] == "leaf0" and set(event["block_val_acc"]) == {"b3", "b4"}
    assert grow(tree, train, "cpu", 2, tmp_path, max_nodes=7, limit_batches=6,
                val_loader=val) is not None  # leaf1 is still never picked
    assert tree.graph.nodes["leaf1"].kind == "leaf"


def test_train_run_grows_a_tree_and_logs_each_split(tmp_path, capsys):
    d = run(parse_args(["--dataset", "synthetic_cifar", "--model", "tree", "--device", "cpu",
                        "--epochs", "3", "--width", "4", "--max-nodes", "5",
                        "--route-acc-min", "0.9", "--runs-dir", str(tmp_path)]))["run_dir"]
    events = [json.loads(s) for s in (d / "growth.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in events] == [1, 2]
    for e in events:
        assert {"groups", "block", "block_val_acc", "fallback"} <= set(e)
        assert e["block"] in e["block_val_acc"]
    assert sorted(c for g in events[0]["groups"] for c in g) == list(range(10))
    assert (d / "graphs" / "epoch_002.json").exists()
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
