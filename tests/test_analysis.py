import pytest
import torch

from beamroot.analysis import collect_stats, node_depths, purity
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


def force(model, name, probs):
    out = model.nodes[name].out
    with torch.no_grad():
        out.weight.zero_()
        out.bias.copy_(torch.tensor(probs).log())


def test_purity_is_share_of_dominant_class():
    assert purity([6.0, 2.0, 2.0]) == (0, pytest.approx(0.6))


def test_purity_with_class_groups_uses_group_mass():
    groups = {0: "animal", 1: "animal", 2: "vehicle"}
    assert purity([3.0, 3.0, 2.0], groups) == ("animal", pytest.approx(0.75))


def test_purity_of_empty_node_is_none():
    assert purity([0.0, 0.0]) == (None, 0.0)


def test_node_depth_is_longest_path_from_root():
    g = diamond()
    d = node_depths(g)
    assert d == {"root": 0, "a": 1, "b": 1, "la": 2, "shared": 2, "lb": 2}
    g.split("shared")  # shared becomes a router, children at depth 3
    d = node_depths(g)
    assert d["shared"] == 2 and all(d[c] == 3 for c in g.children("shared"))


def test_node_depth_uses_longest_path_when_parents_differ_in_depth():
    g = RouterGraph()
    g.add_node("root", "router")
    g.add_node("r", "router", parents=["root"])
    g.add_node("x", "leaf", parents=["root", "r"])
    g.add_node("y", "leaf", parents=["r"])
    assert node_depths(g)["x"] == 2


def test_collect_stats_class_mass_is_reach_weighted_and_edges_are_mean_probs():
    torch.manual_seed(0)
    model = DAGRouterModel(graph=diamond()).eval()
    force(model, "root", [0.75, 0.25])
    force(model, "a", [0.5, 0.5])
    force(model, "b", [0.2, 0.8])
    x = torch.randn(4, 1, 28, 28)
    y = torch.tensor([0, 0, 1, 2])
    stats = collect_stats(model, [(x, y)], "cpu", num_classes=10)
    mass = stats["class_mass"]
    # root sees every sample; a sees 0.75 of each; shared sees 0.375 + 0.05 of each
    assert mass["root"][0] == pytest.approx(2.0)
    assert mass["a"][0] == pytest.approx(1.5)
    assert mass["a"][1] == pytest.approx(0.75)
    assert mass["shared"][0] == pytest.approx(2 * (0.375 + 0.05))
    assert mass["lb"][2] == pytest.approx(0.2)
    assert stats["edge_prob"][("root", "a")] == pytest.approx(0.75)
    assert stats["edge_prob"][("b", "lb")] == pytest.approx(0.8)
    assert stats["edge_prob"][("a", "shared")] == pytest.approx(0.5)


def test_cifar10_groups_split_vehicles_from_animals():
    from beamroot.data import CIFAR10_CLASSES, CIFAR10_GROUPS
    vehicles = {CIFAR10_CLASSES[i] for i, g in CIFAR10_GROUPS.items() if g == "vehicle"}
    assert vehicles == {"airplane", "automobile", "ship", "truck"}
    assert sum(g == "animal" for g in CIFAR10_GROUPS.values()) == 6
    mass = [0.0] * 10
    mass[0], mass[8], mass[3] = 3.0, 3.0, 2.0  # two vehicle classes, one animal
    assert purity(mass, CIFAR10_GROUPS) == ("vehicle", pytest.approx(0.75))


def test_collect_stats_router_entropy_is_mean_per_image_entropy():
    import math
    model = DAGRouterModel(graph=diamond()).eval()
    force(model, "root", [0.75, 0.25])
    force(model, "a", [0.5, 0.5])
    force(model, "b", [0.2, 0.8])
    batches = [(torch.randn(3, 1, 28, 28), torch.tensor([0, 1, 2])),
               (torch.randn(1, 1, 28, 28), torch.tensor([3]))]
    ent = collect_stats(model, batches, "cpu", num_classes=10)["entropy"]
    h = lambda ps: -sum(p * math.log(p) for p in ps)  # noqa: E731
    assert set(ent) == {"root", "a", "b"}
    assert ent["root"] == pytest.approx(h([0.75, 0.25]))
    assert ent["a"] == pytest.approx(math.log(2))
    assert ent["b"] == pytest.approx(h([0.2, 0.8]))


def flow_mass():
    # 4 classes; root mass 100 each
    return {"root": [100, 100, 100, 100], "a": [50, 10, 9, 0], "b": [50, 90, 91, 100],
            "la": [50, 0, 0, 0], "shared": [0, 10, 9, 0], "lb": [0, 90, 91, 100]}


def test_class_flow_tree_with_shared_leaf_prints_subtree_once():
    from beamroot.analysis import class_flow
    lines = class_flow(diamond(), flow_mass()).splitlines()
    assert lines[0] == "root  reach 100%  {all classes}"
    # a holds exactly 50% / 10% of classes 0 / 1 -> in set; class 2 at 9% -> out
    assert lines[1] == "├─ a  reach 17%  {0,1}"
    assert lines[2] == "│  ├─ la  reach 12%  0 (100%)  ← pure"
    assert lines[3] == "│  └─ shared  reach 5%  1 (53%), 2 (47%)"
    assert lines[4].startswith("└─ b  reach 83%  {")
    assert lines[5] == "   ├─ → shared  (shared, see above)"
    assert lines[6].startswith("   └─ lb  reach 70%")
    assert len(lines) == 7


def test_class_flow_leaf_thresholds_are_inclusive():
    from beamroot.analysis import class_flow
    g = RouterGraph()
    g.add_node("root", "router")
    g.add_node("x", "leaf", parents=["root"])
    g.add_node("y", "leaf", parents=["root"])
    mass = {"root": [100, 100], "x": [95, 5], "y": [5, 95]}
    lines = class_flow(g, mass).splitlines()
    assert lines[1] == "├─ x  reach 50%  0 (95%), 1 (5%)  ← pure"
    assert "← pure" in lines[2] and "0 (5%)" in lines[2]


def test_class_flow_uses_class_names_when_given():
    from beamroot.analysis import class_flow
    out = class_flow(diamond(), flow_mass(), class_names=["w", "x", "y", "z"])
    assert "{w,x}" in out and "w (100%)" in out
