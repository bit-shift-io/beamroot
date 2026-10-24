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


def set_routing(model, name, probs):
    """Force a router to emit fixed probabilities regardless of input."""
    out = model.nodes[name].out
    with torch.no_grad():
        out.weight.zero_()
        out.bias.copy_(torch.tensor(probs).log())


def test_diamond_reach_probabilities():
    torch.manual_seed(0)
    model = DAGRouterModel(graph=diamond())
    set_routing(model, "root", [0.3, 0.7])   # -> a, b
    set_routing(model, "a", [0.4, 0.6])      # -> la, shared
    set_routing(model, "b", [0.5, 0.5])      # -> shared, lb
    reach = {n: r.exp() for n, r in model.route(torch.randn(4, 1, 28, 28))["log_reach"].items()}
    expected = {"root": 1.0, "a": 0.3, "b": 0.7, "la": 0.12,
                "shared": 0.3 * 0.6 + 0.7 * 0.5, "lb": 0.35}
    for n, p in expected.items():
        assert torch.allclose(reach[n], torch.full((4,), p), atol=1e-5), n


def multi_parent_dag():
    g = diamond()
    g.add_edge("a", "b")  # b reached via root and via a
    return g


def test_leaf_reach_sums_to_one_on_multi_parent_dag():
    torch.manual_seed(0)
    model = DAGRouterModel(graph=multi_parent_dag())
    reach = model.route(torch.randn(8, 1, 28, 28))["log_reach"]
    total = sum(reach[n].exp() for n in model.graph.leaves())
    assert torch.allclose(total, torch.ones(8), atol=1e-5)


def test_output_is_valid_distribution():
    torch.manual_seed(0)
    model = DAGRouterModel(graph=multi_parent_dag()).eval()
    probs = model(torch.randn(8, 1, 28, 28)).exp()
    assert probs.shape == (8, 10)
    assert (probs >= 0).all()
    assert torch.allclose(probs.sum(1), torch.ones(8), atol=1e-5)


def test_gradients_reach_stem_and_every_router():
    torch.manual_seed(0)
    model = DAGRouterModel(graph=multi_parent_dag())
    loss = torch.nn.functional.nll_loss(model(torch.randn(8, 1, 28, 28)),
                                        torch.randint(0, 10, (8,)))
    loss.backward()
    assert model.stem[0].weight.grad.abs().sum() > 0
    for name in model.graph.routers():
        for p in model.nodes[name].parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_new_nodes_are_registered_parameters():
    model = DAGRouterModel()
    model.graph.add_node("extra", "leaf", parents=["root"])
    model.nodes["extra"] = model.build_node("extra")
    names = {n for n, _ in model.named_parameters()}
    assert any(n.startswith("nodes.extra.") for n in names)
    assert any(k.startswith("nodes.extra.") for k in model.state_dict())


def test_checkpoint_round_trip_restores_identical_outputs(tmp_path):
    torch.manual_seed(0)
    model = DAGRouterModel(graph=multi_parent_dag()).eval()
    x = torch.randn(4, 1, 28, 28)
    torch.save(model.checkpoint(), tmp_path / "model.pt")
    torch.manual_seed(123)
    restored = DAGRouterModel.from_checkpoint(torch.load(tmp_path / "model.pt")).eval()
    assert restored.graph.edges == model.graph.edges
    assert restored.graph.parents("b") == ["root", "a"]
    assert torch.equal(restored(x), model(x))


def test_train_cli_dag_model_trains_and_checkpoint_restores(tmp_path):
    from beamroot.train import parse_args, run
    res = run(parse_args(["--dataset", "synthetic", "--model", "dag", "--device", "cpu",
                          "--epochs", "3", "--grow-every", "0", "--runs-dir", str(tmp_path)]))
    assert res["loss"] < res["first_loss"]
    assert res["test_acc"] > 0.9
    ckpt = torch.load(res["run_dir"] / "model.pt")
    restored = DAGRouterModel.from_checkpoint(ckpt)
    assert sorted(restored.graph.nodes) == ["leaf0", "leaf1", "root"]
