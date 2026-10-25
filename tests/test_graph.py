import pytest

from beamroot.models.graph import RouterGraph


def diamond():
    g = RouterGraph()
    g.add_node("root", "router")
    g.add_node("a", "router", parents=["root"])
    g.add_node("b", "router", parents=["root"])
    g.add_node("la", "leaf", parents=["a"])
    g.add_node("shared", "leaf", parents=["a", "b"])
    g.add_node("lb", "leaf", parents=["b"])
    return g


def test_nodes_hold_parent_lists():
    g = diamond()
    assert g.parents("shared") == ["a", "b"]
    assert g.parents("root") == []
    assert g.children("a") == ["la", "shared"]


def test_rejects_cycles():
    g = diamond()
    with pytest.raises(ValueError, match="cycle"):
        g.add_edge("a", "root")
    with pytest.raises(ValueError, match="cycle"):
        g.add_edge("a", "a")
    assert ("a", "root") not in g.edges
    assert g.parents("root") == []


def test_leaf_cannot_have_children():
    g = diamond()
    with pytest.raises(ValueError):
        g.add_node("x", "leaf", parents=["shared"])


def test_topological_order_valid_on_multi_parent_dag():
    g = diamond()
    g.add_edge("a", "b")  # b now has two parents: root and a
    order = g.topological_order()
    assert sorted(order) == sorted(g.nodes)
    pos = {n: i for i, n in enumerate(order)}
    for p, c in g.edges:
        assert pos[p] < pos[c]
    assert g.root == "root"
    assert g.leaves() == [n for n in order if n in {"la", "shared", "lb"}]


def test_remove_node_drops_its_edges():
    g = diamond()
    g.remove_node("a")
    assert "a" not in g.nodes
    assert g.parents("shared") == ["b"]
    assert all("a" not in e for e in g.edges)


def test_dict_round_trip_preserves_structure():
    g = diamond()
    g2 = RouterGraph.from_dict(g.to_dict())
    assert g2.edges == g.edges
    assert {n: g2.parents(n) for n in g2.nodes} == {n: g.parents(n) for n in g.nodes}
    assert {n: v.kind for n, v in g2.nodes.items()} == {n: v.kind for n, v in g.nodes.items()}


def test_split_with_a_link_gives_the_linked_leaf_a_second_parent():
    g = diamond()
    new = g.split("la", 1)
    g.add_edge("la", "lb")  # the split's second group links to existing leaf lb
    assert g.children("la") == [new[0], "lb"]
    assert g.parents("lb") == ["b", "la"]
    order = g.topological_order()
    assert sorted(order) == sorted(g.nodes)
    pos = {n: i for i, n in enumerate(order)}
    assert all(pos[p] < pos[c] for p, c in g.edges)
    g2 = RouterGraph.from_dict(g.to_dict())
    assert g2.edges == g.edges and g2.parents("lb") == ["b", "la"]
    assert g2.topological_order() == order
