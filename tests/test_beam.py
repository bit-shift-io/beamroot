import torch

from beamroot.beam import beam_search
from beamroot.models.dag import DAGRouterModel
from beamroot.models.graph import RouterGraph


def tree():
    """root -> r1, r2; r1 -> l0, l1; r2 -> l2, l3."""
    g = RouterGraph()
    g.add_node("root", "router")
    g.add_node("r1", "router", parents=["root"])
    g.add_node("r2", "router", parents=["root"])
    for name, parent in [("l0", "r1"), ("l1", "r1"), ("l2", "r2"), ("l3", "r2")]:
        g.add_node(name, "leaf", parents=[parent])
    return g


def set_routing(model, name, probs):
    """Force a router to emit fixed probabilities regardless of input."""
    out = model.nodes[name].out
    with torch.no_grad():
        out.weight.zero_()
        out.bias.copy_(torch.tensor(probs).log())


def set_leaf_class(model, name, cls, confidence=0.9):
    """Force a leaf to predict `cls` with the given probability regardless of input."""
    last = model.nodes[name].net[-1]
    n = last.out_features
    probs = torch.full((n,), (1 - confidence) / (n - 1))
    probs[cls] = confidence
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(probs.log())


def fixed_tree_model():
    torch.manual_seed(0)
    model = DAGRouterModel(graph=tree()).eval()
    set_routing(model, "root", [0.6, 0.4])
    set_routing(model, "r1", [0.5, 0.5])   # l0, l1 = 0.30 each
    set_routing(model, "r2", [0.9, 0.1])   # l2 = 0.36, l3 = 0.04
    for i, leaf in enumerate(["l0", "l1", "l2", "l3"]):
        set_leaf_class(model, leaf, i)
    return model


def test_beam_one_follows_most_probable_child_greedily():
    model = fixed_tree_model()
    [res] = beam_search(model, torch.randn(1, 1, 28, 28), beam=1)
    # Greedy: root picks r1 (0.6), r1 then picks l0 (ties broken by child order).
    assert [p.leaf for p in res.paths] == ["l0"]
    assert res.paths[0].nodes == ["root", "r1", "l0"]
    assert res.pred == 0


def test_beam_n_returns_paths_in_descending_probability():
    model = fixed_tree_model()
    [res] = beam_search(model, torch.randn(1, 1, 28, 28), beam=4)
    # Nothing is pruned; l2 (0.36) beats l0/l1 (0.30) and l3 (0.04).
    assert [p.leaf for p in res.paths] == ["l2", "l0", "l1", "l3"]
    assert [round(p.prob, 4) for p in res.paths] == [0.36, 0.3, 0.3, 0.04]
    assert res.paths[0].nodes == ["root", "r2", "l2"]
    assert [p.pred for p in res.paths] == [2, 0, 1, 3]


def test_beam_prunes_to_n_most_probable_partial_paths():
    model = fixed_tree_model()
    [res] = beam_search(model, torch.randn(1, 1, 28, 28), beam=2)
    assert [p.leaf for p in res.paths] == ["l2", "l0"]


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


class CountingCalls:
    """Records which node modules run, via forward hooks (no behaviour change)."""

    def __init__(self, model):
        self.calls = []
        for name, module in model.nodes.items():
            module.register_forward_hook(lambda *_, n=name: self.calls.append(n))


def test_diamond_shared_node_evaluated_once_with_summed_probability():
    torch.manual_seed(0)
    model = DAGRouterModel(graph=diamond()).eval()
    set_routing(model, "root", [0.3, 0.7])   # -> a, b
    set_routing(model, "a", [0.4, 0.6])      # -> la, shared
    set_routing(model, "b", [0.5, 0.5])      # -> shared, lb
    counter = CountingCalls(model)
    [res] = beam_search(model, torch.randn(1, 1, 28, 28), beam=3)
    shared = next(p for p in res.paths if p.leaf == "shared")
    assert abs(shared.prob - (0.3 * 0.6 + 0.7 * 0.5)) < 1e-5
    # Highest-probability incoming path is via b (0.35 > 0.18).
    assert shared.nodes == ["root", "b", "shared"]
    assert abs(shared.path_prob - 0.35) < 1e-5
    assert counter.calls.count("shared") == 1
    assert res.paths[0].leaf == "shared"


def multi_parent_dag():
    g = diamond()
    g.add_edge("a", "b")  # b reached via root and via a
    return g


def test_beam_at_least_leaf_count_matches_soft_routing():
    for graph in (tree(), diamond(), multi_parent_dag()):
        torch.manual_seed(0)
        model = DAGRouterModel(graph=graph).eval()
        x = torch.randn(6, 1, 28, 28)
        soft = model(x).exp()
        results = beam_search(model, x, beam=len(graph.leaves()))
        beam_probs = torch.stack([r.probs for r in results])
        assert torch.allclose(beam_probs, soft, atol=1e-5)
        assert [r.pred for r in results] == soft.argmax(1).tolist()


def test_nodes_evaluated_counts_routers_expanded_and_leaves_kept():
    model = fixed_tree_model()
    [one] = beam_search(model, torch.randn(1, 1, 28, 28), beam=1)
    [two] = beam_search(model, torch.randn(1, 1, 28, 28), beam=2)
    [four] = beam_search(model, torch.randn(1, 1, 28, 28), beam=4)
    assert one.nodes_evaluated == 3    # root, r1, l0
    assert two.nodes_evaluated == 5    # root, r1, r2, l2, l0
    assert four.nodes_evaluated == 7   # every node


def tiny_dag_run(tmp_path):
    from beamroot.train import parse_args, run
    return run(parse_args(["--dataset", "synthetic", "--model", "dag", "--device", "cpu",
                           "--epochs", "1", "--limit-batches", "2",
                           "--runs-dir", str(tmp_path)]))["run_dir"]


def test_evaluate_cli_prints_table_of_modes(tmp_path, capsys):
    from beamroot.evaluate import main
    run_dir = tiny_dag_run(tmp_path)
    rows = main([str(run_dir), "--device", "cpu"])
    assert [r["mode"] for r in rows] == ["beam 1", "beam 3", "beam 5", "soft"]
    by_mode = {r["mode"]: r for r in rows}
    # Default graph has 2 leaves, so beam 3 and 5 equal soft routing.
    assert by_mode["beam 3"]["accuracy"] == by_mode["soft"]["accuracy"]
    assert by_mode["beam 1"]["mean_nodes"] == 2
    assert by_mode["soft"]["mean_nodes"] == 3
    out = capsys.readouterr().out
    assert "mode" in out and "accuracy" in out and "mean nodes evaluated" in out
    for mode in by_mode:
        assert mode in out


def test_predict_cli_prints_top_n_paths_per_sample(tmp_path, capsys):
    from beamroot.predict import main
    run_dir = tiny_dag_run(tmp_path)
    capsys.readouterr()
    results = main([str(run_dir), "--beam", "2", "--samples", "3", "--device", "cpu"])
    assert len(results) == 3
    out = capsys.readouterr().out
    assert out.count("true label") == 3
    for label, res in results:
        assert len(res.paths) == 2
        for p in res.paths:
            assert " -> ".join(p.nodes) in out
            assert f"{p.prob:.4f}" in out
            assert f"class {p.pred}" in out
