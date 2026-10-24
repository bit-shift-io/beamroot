import json

import pytest
import torch

from beamroot.models.dag import DAGRouterModel
from beamroot.models.graph import RouterGraph


def cousins():
    """root -> a, b (routers); a -> la1, la2; b -> lb1, lb2. Leaves la1 and lb1 are cousins."""
    g = RouterGraph()
    g.add_node("root", "router")
    g.add_node("a", "router", parents=["root"])
    g.add_node("b", "router", parents=["root"])
    for p in ("a", "b"):
        g.add_node(f"l{p}1", "leaf", parents=[p])
        g.add_node(f"l{p}2", "leaf", parents=[p])
    return g


def test_merge_leaves_combines_parents_and_keeps_child_order():
    g = cousins()
    removed = g.merge("la1", "lb1")
    assert removed == ["lb1"]
    assert "lb1" not in g.nodes
    assert g.parents("la1") == ["a", "b"]
    assert g.children("b") == ["la1", "lb2"]  # same output column now feeds the survivor
    assert sorted(g.topological_order()) == sorted(g.nodes)
    assert len(g.edges) == len(set(g.edges))



def deep():
    """root -> a, b; a -> a1, la; b -> b1, lb; a1 -> x1, x2; b1 -> y1, y2 (a1, b1 cousin routers)."""
    g = RouterGraph()
    g.add_node("root", "router")
    for top, sub, leaf, kids in [("a", "a1", "la", ("x1", "x2")), ("b", "b1", "lb", ("y1", "y2"))]:
        g.add_node(top, "router", parents=["root"])
        g.add_node(sub, "router", parents=[top])
        g.add_node(leaf, "leaf", parents=[top])
        for k in kids:
            g.add_node(k, "leaf", parents=[sub])
    return g


def test_merge_routers_prunes_the_dropped_subgraph():
    g = deep()
    removed = g.merge("a1", "b1")
    assert sorted(removed) == ["b1", "y1", "y2"]
    assert g.parents("a1") == ["a", "b"]
    assert g.children("b") == ["a1", "lb"]
    assert g.children("a1") == ["x1", "x2"]
    assert sorted(g.topological_order()) == sorted(g.nodes)


def test_prune_keeps_nodes_still_reachable_through_another_parent():
    g = deep()
    g.add_edge("a1", "y1")  # y1 now also hangs off a1
    removed = g.merge("la", "lb")
    assert removed == ["lb"]
    g2 = deep()
    g2.add_edge("a1", "y1")
    g2.remove_node("b1")
    assert g2.prune_unreachable("root") == ["y2"]
    assert g2.parents("y1") == ["a1"]


def test_merge_refuses_ancestor_pairs_shared_parents_and_mixed_kinds():
    g = deep()
    for keep, drop in [("a", "a1"), ("a1", "a"), ("x1", "x2"), ("a1", "la"), ("a", "b")]:
        with pytest.raises(ValueError):
            g.merge(keep, drop)
    assert g.to_dict() == deep().to_dict()


def batch(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(n, 1, 28, 28, generator=g), torch.randint(0, 10, (n,), generator=g))]


def test_identical_cousin_leaves_are_the_closest_candidate_and_merge_without_changing_output():
    from beamroot.growth import merge_candidates, merge_nodes
    torch.manual_seed(0)
    model = DAGRouterModel(graph=cousins()).eval()
    model.nodes["lb1"].load_state_dict(model.nodes["la1"].state_dict())
    data = batch()
    cands = merge_candidates(model, data, "cpu")
    kl, keep, drop = cands[0]
    assert (keep, drop) == ("la1", "lb1")
    assert kl == pytest.approx(0.0, abs=1e-6)
    pairs = {(k, d) for _, k, d in cands}
    assert pairs == {("la1", "lb1"), ("la1", "lb2"), ("la2", "lb1"), ("la2", "lb2")}
    before = model(data[0][0])
    removed = merge_nodes(model, keep, drop)
    assert removed == ["lb1"]
    assert set(model.nodes) == set(model.graph.nodes)
    assert model.graph.parents("la1") == ["a", "b"]
    assert torch.allclose(model(data[0][0]), before, atol=1e-6)


def test_router_merge_with_matching_subgraphs_keeps_soft_output():
    from beamroot.growth import merge_candidates, merge_nodes
    torch.manual_seed(0)
    model = DAGRouterModel(graph=deep()).eval()
    for src, dst in [("a1", "b1"), ("x1", "y1"), ("x2", "y2")]:
        model.nodes[dst].load_state_dict(model.nodes[src].state_dict())
    data = batch()
    kl = {(k, d): v for v, k, d in merge_candidates(model, data, "cpu")}
    assert kl[("a1", "b1")] == pytest.approx(0.0, abs=1e-6)
    assert ("a", "b") not in kl and ("la", "a1") not in kl
    before = model(data[0][0])
    assert sorted(merge_nodes(model, "a1", "b1")) == ["b1", "y1", "y2"]
    assert set(model.nodes) == set(model.graph.nodes)
    assert torch.allclose(model(data[0][0]), before, atol=1e-5)


def test_merge_step_merges_pairs_below_threshold_and_logs_events(tmp_path):
    from beamroot.growth import merge_step
    torch.manual_seed(0)
    model = DAGRouterModel(graph=cousins())
    model.nodes["lb1"].load_state_dict(model.nodes["la1"].state_dict())
    events = merge_step(model, batch(), "cpu", epoch=3, run_dir=tmp_path, threshold=1e-3)
    assert len(events) == 1
    ev = events[0]
    assert ev["merge"] == ["la1", "lb1"] and ev["removed"] == ["lb1"]
    assert ev["epoch"] == 3 and ev["num_nodes"] == 6 and ev["threshold"] == 1e-3
    assert ev["kl"] == pytest.approx(0.0, abs=1e-6)
    lines = [json.loads(s) for s in (tmp_path / "growth.jsonl").read_text().splitlines()]
    assert lines == events
    snap = json.loads((tmp_path / "graphs" / "epoch_003.json").read_text())
    assert RouterGraph.from_dict(snap).parents("la1") == ["a", "b"]


def test_merge_step_logs_when_no_pair_meets_threshold(tmp_path):
    from beamroot.growth import merge_step
    torch.manual_seed(0)
    model = DAGRouterModel(graph=cousins())
    assert merge_step(model, batch(), "cpu", epoch=1, run_dir=tmp_path, threshold=1e-9) == []
    (line,) = [json.loads(s) for s in (tmp_path / "growth.jsonl").read_text().splitlines()]
    assert line["merge"] is None and line["epoch"] == 1 and line["threshold"] == 1e-9
    assert set(line["candidates"]) == {"la1+lb1", "la1+lb2", "la2+lb1", "la2+lb2"}
    assert all(v > 1e-9 for v in line["candidates"].values())
    assert len(model.graph.nodes) == 7


def test_pruned_params_leave_the_optimiser_and_survivors_keep_state():
    from beamroot.growth import merge_nodes, rebuild_optimizer
    torch.manual_seed(0)
    model = DAGRouterModel(graph=deep())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = batch()[0]
    torch.nn.functional.nll_loss(model(x), y).backward()
    opt.step()
    gone = [p for n in ("b1", "y1", "y2") for p in model.nodes[n].parameters()]
    kept = next(model.nodes["a1"].parameters())
    kept_state = {k: v.clone() for k, v in opt.state[kept].items()}
    merge_nodes(model, "a1", "b1")
    opt = rebuild_optimizer(model, opt)
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert ids == {id(p) for p in model.parameters()}
    assert not ids & {id(p) for p in gone}
    for k, v in kept_state.items():
        assert torch.equal(opt.state[kept][k], v)


def test_train_cli_with_merge_logs_a_merge_check_every_growth_step(tmp_path):
    from beamroot.train import parse_args, run
    res = run(parse_args(["--dataset", "synthetic", "--model", "dag", "--device", "cpu",
                          "--epochs", "4", "--max-nodes", "9", "--merge",
                          "--merge-threshold", "1e9", "--no-report", "--runs-dir", str(tmp_path)]))
    run_dir = res["run_dir"]
    events = [json.loads(s) for s in (run_dir / "growth.jsonl").read_text().splitlines()]
    merge_epochs = {e["epoch"] for e in events if "merge" in e}
    assert merge_epochs == {1, 2, 3}
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    model = DAGRouterModel.from_checkpoint(ckpt)
    assert sorted(model.graph.topological_order()) == sorted(model.graph.nodes)
    assert json.loads((run_dir / "config.json").read_text())["merge_threshold"] == 1e9


def test_node_labels_mark_merged_nodes_with_their_parents():
    from beamroot.report import node_labels
    g = cousins()
    g.merge("la1", "lb1")
    labels = node_labels(g, {n: [1.0, 0.0] for n in g.nodes})
    assert "merged" in labels["la1"] and "a+b" in labels["la1"]
    assert "merged" not in labels["la2"]


def test_report_lists_parents_of_merged_nodes(tmp_path):
    from beamroot.growth import merge_nodes
    from beamroot.report import main
    from beamroot.train import parse_args, run
    run_dir = run(parse_args(["--dataset", "synthetic", "--model", "dag", "--device", "cpu",
                              "--epochs", "1", "--limit-batches", "1", "--grow-every", "0",
                              "--no-report", "--runs-dir", str(tmp_path)]))["run_dir"]
    model = DAGRouterModel(graph=cousins())
    merge_nodes(model, "la1", "lb1")
    torch.save(model.checkpoint(), run_dir / "model.pt")
    main([str(run_dir), "--device", "cpu", "--samples", "2", "--limit-batches", "1"])
    html = (run_dir / "report.html").read_text()
    assert "1 merged node" in html
    assert "<td>la1</td><td>leaf (merged)</td><td>a, b</td>" in html
