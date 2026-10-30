"""Beam search over the Router graph (see CONTEXT.md, ADR 0001).

The beam is a frontier of nodes reached but not yet expanded, each carrying
the combined probability of every retained path into it. A step expands each
router in the frontier that no other frontier node can still reach (so all
its incoming probability has arrived), then keeps the `beam` most probable
frontier nodes. Paths reaching the same node add, and each node is evaluated
at most once. Search ends when only leaves remain; those leaves are then
evaluated and mixed, weighted by their renormalised probability.

With merges, "the path" to a node is ambiguous: each result reports the
highest-probability incoming path alongside the combined probability.
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class BeamPath:
    leaf: str
    nodes: list[str]   # highest-probability path from root to leaf
    prob: float        # combined probability of all retained paths into the leaf
    path_prob: float   # probability of `nodes` alone
    pred: int          # the leaf's predicted class


@dataclass
class BeamResult:
    paths: list[BeamPath]   # descending by combined probability
    probs: torch.Tensor     # (C,) class distribution from the retained leaves
    nodes_evaluated: int
    evaluated: list[str] = field(default_factory=list)  # names of evaluated nodes

    @property
    def pred(self) -> int:
        return int(self.probs.argmax())


@dataclass
class _Entry:
    prob: float
    best_path: list[str]
    best_prob: float


def beam_search(model, x: torch.Tensor, beam: int) -> list[BeamResult]:
    """Beam search for each image in batch `x`; the Stem runs once per batch."""
    if beam < 1:
        raise ValueError("beam must be >= 1")
    graph = model.graph
    descendants = {n: graph.descendants(n) for n in graph.nodes}
    with torch.no_grad():
        feats = model.features(x)
        return [_search_one(model, feats[i:i + 1], beam, descendants)
                for i in range(feats.shape[0])]


def _search_one(model, feats, beam, descendants) -> BeamResult:
    graph = model.graph
    root = graph.root
    frontier = {root: _Entry(1.0, [root], 1.0)}
    evaluated = []
    while True:
        ready = [n for n in frontier if graph.nodes[n].kind == "router"
                 and not any(n in descendants[m] for m in frontier if m != n)]
        if not ready:
            break
        for name in ready:
            entry = frontier.pop(name)
            route = model.nodes[name](feats)[0].exp().tolist()
            evaluated.append(name)
            for child, p in zip(graph.children(name), route):
                path_prob = entry.best_prob * p
                if child not in frontier:
                    frontier[child] = _Entry(0.0, [], -1.0)
                target = frontier[child]
                target.prob += entry.prob * p
                if path_prob > target.best_prob:
                    target.best_path, target.best_prob = entry.best_path + [child], path_prob
        kept = sorted(frontier.items(), key=lambda kv: -kv[1].prob)[:beam]
        frontier = dict(kept)

    paths, mix = [], 0
    total = sum(e.prob for e in frontier.values())
    for leaf, entry in frontier.items():
        leaf_probs = F.softmax(model.nodes[leaf](feats)[0], dim=0)
        evaluated.append(leaf)
        mix = mix + (entry.prob / total) * leaf_probs
        paths.append(BeamPath(leaf, entry.best_path, entry.prob, entry.best_prob,
                              int(leaf_probs.argmax())))
    return BeamResult(paths, mix, len(evaluated), evaluated)
