"""Per-node class mix, purity, edge routing probabilities and depth for a trained DAG model.

Class mix is reach-weighted: a sample contributes to a node in proportion to
the soft probability that it reaches that node, so the numbers stay meaningful
when a node has several parents.
"""
from itertools import islice

import torch

from beamroot.losses import routing_entropy


def node_depths(graph) -> dict[str, int]:
    """Depth of each node = length of the longest path from the root."""
    depth = {}
    for name in graph.topological_order():
        parents = graph.parents(name)
        depth[name] = 1 + max(depth[p] for p in parents) if parents else 0
    return depth


def purity(class_mass, groups: dict | None = None):
    """(dominant class or group, its share of the mass). `groups` maps class index -> group label.

    An empty node (no mass) gives (None, 0.0).
    """
    totals: dict = {}
    for cls, m in enumerate(class_mass):
        key = groups[cls] if groups is not None else cls
        totals[key] = totals.get(key, 0.0) + float(m)
    total = sum(totals.values())
    if total <= 0:
        return None, 0.0
    best = max(totals, key=totals.get)
    return best, totals[best] / total


def collect_stats(model, loader, device, num_classes, limit_batches=None) -> dict:
    """One pass over `loader`.

    Returns {"class_mass": {node: [C floats]}, "edge_prob": {(parent, child): float},
    "entropy": {router: float}}. class_mass sums reach(node) per true class; edge_prob is
    the reach-weighted mean routing probability of the edge; entropy is each router's
    routing entropy (mean per-image entropy of its child probabilities, not reach-weighted).
    """
    graph = model.graph
    mass = {n: torch.zeros(num_classes, dtype=torch.float64) for n in graph.nodes}
    edge_num = {e: 0.0 for e in graph.edges}
    edge_den = {e: 0.0 for e in graph.edges}
    ent_sum, count = {}, 0
    model.eval()
    with torch.no_grad():
        for x, y in islice(loader, limit_batches):
            out = model.route(x.to(device))
            onehot = torch.nn.functional.one_hot(y, num_classes).double()
            reach = {n: lr.exp().cpu().double() for n, lr in out["log_reach"].items()}
            for n, r in reach.items():
                mass[n] += r @ onehot
            count += len(y)
            for parent, lr in out["log_route"].items():
                ent_sum[parent] = ent_sum.get(parent, 0.0) + float(routing_entropy(lr)) * len(y)
                probs = lr.exp().cpu().double()
                for i, child in enumerate(graph.children(parent)):
                    edge_num[(parent, child)] += float((reach[parent] * probs[:, i]).sum())
                    edge_den[(parent, child)] += float(reach[parent].sum())
    return {"class_mass": {n: m.tolist() for n, m in mass.items()},
            "edge_prob": {e: edge_num[e] / edge_den[e] if edge_den[e] > 0 else 0.0
                          for e in edge_num},
            "entropy": {r: e / count for r, e in ent_sum.items()}}


def class_flow(graph, class_mass, class_names=None) -> str:
    """Text tree of which classes reach each node, with reach (node mass / root mass).

    Routers list classes with >= 10% of that class's root mass reaching them; leaves list
    classes with >= 5% of their own mass and are marked pure at >= 95%. A node met under a
    second parent prints one "(shared, see above)" line and no subtree.
    """
    root = graph.topological_order()[0]
    name = (lambda c: str(class_names[c])) if class_names else str
    root_mass = [float(m) for m in class_mass[root]]
    root_total = sum(root_mass)

    def describe(n):
        mass = [float(m) for m in class_mass[n]]
        total = sum(mass)
        reach = total / root_total if root_total > 0 else 0.0
        head = f"{n}  reach {reach * 100:.0f}%"
        if n == root:
            return f"{head}  {{all classes}}"
        if graph.nodes[n].kind == "router":
            share = {c: m / root_mass[c] for c, m in enumerate(mass) if root_mass[c] > 0}
            kept = sorted((c for c, s in share.items() if s >= 0.1 - 1e-12),
                          key=lambda c: -share[c])
            return f"{head}  {{{','.join(name(c) for c in kept)}}}"
        if total <= 0:
            return head
        parts = [f"{name(c)} ({m / total * 100:.0f}%)"
                 for c, m in sorted(enumerate(mass), key=lambda cm: -cm[1])
                 if m / total >= 0.05 - 1e-12]
        pure = "  ← pure" if max(mass) / total >= 0.95 - 1e-12 else ""
        return f"{head}  {', '.join(parts)}{pure}"

    lines, seen = [describe(root)], {root}

    def walk(n, prefix):
        kids = graph.children(n)
        for i, c in enumerate(kids):
            last = i == len(kids) - 1
            branch, pad = ("└─ ", "   ") if last else ("├─ ", "│  ")
            if c in seen:
                lines.append(f"{prefix}{branch}→ {c}  (shared, see above)")
                continue
            seen.add(c)
            lines.append(f"{prefix}{branch}{describe(c)}")
            walk(c, prefix + pad)

    walk(root, "")
    return "\n".join(lines)
