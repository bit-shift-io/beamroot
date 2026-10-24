"""Growth steps: split the worst leaf of a DAG router model (see CONTEXT.md: Growth step, Split)."""
import copy
import json
from itertools import islice
from pathlib import Path

import torch

from beamroot.analysis import node_depths


def leaf_losses(model, loader, device, limit_batches=None) -> dict[str, float]:
    """Per-leaf reach-weighted loss: mean over samples of reach(leaf) * NLL(leaf, label).

    Measured in eval mode (no dropout), without gradients.
    """
    was_training = model.training
    model.eval()
    sums, total = {}, 0
    with torch.no_grad():
        for x, y in islice(loader, limit_batches):
            out = model.route(x.to(device))
            y = y.to(device)
            for name, lp in out["leaf_log_probs"].items():
                nll = -lp.gather(1, y[:, None]).squeeze(1)
                sums[name] = sums.get(name, 0.0) + (out["log_reach"][name].exp() * nll).sum().item()
            total += len(y)
    model.train(was_training)
    return {n: s / max(total, 1) for n, s in sums.items()}


def choose_leaf(losses: dict[str, float]) -> str:
    """The leaf with the highest reach-weighted loss."""
    return max(losses, key=losses.get)


def split_leaf(model, name: str, noise: float = 1e-3) -> tuple[str, str]:
    """Split leaf `name` of a DAGRouterModel in place.

    The leaf becomes a fresh RouterNode (keeping all its parents); the two new
    leaves start as copies of the old Leaf's weights plus Gaussian noise of std
    `noise`, so the soft output barely changes. Returns the new leaves' names.
    """
    old = model.nodes[name]
    ref = next(old.parameters())
    new = model.graph.split(name)
    model.nodes[name] = model.build_node(name).to(ref.device)
    for n in new:
        leaf = copy.deepcopy(old)
        with torch.no_grad():
            for p in leaf.parameters():
                p.add_(torch.randn_like(p) * noise)
        model.nodes[n] = leaf
    return new


def node_outputs(model, out: dict) -> dict:
    """{node: (B, C)} log class distribution of each node's sub-DAG, given it was reached.

    A leaf's is its own prediction; a router's mixes its children's by its routing probs.
    `out` is the result of `model.route(x)`.
    """
    graph, res = model.graph, {}
    for name in reversed(graph.topological_order()):
        if graph.nodes[name].kind == "leaf":
            res[name] = out["leaf_log_probs"][name]
        else:
            kids = torch.stack([res[c] for c in graph.children(name)], dim=1)  # (B, k, C)
            res[name] = torch.logsumexp(out["log_route"][name][:, :, None] + kids, dim=1)
    return res


def candidate_pairs(graph) -> list[tuple[str, str]]:
    """(keep, drop) pairs that may merge: same depth (longest path), same kind, no shared
    parent, neither an ancestor of the other. `keep` is the earlier one topologically.

    Siblings are excluded: merging them would need one router output column to vanish,
    and every fresh split's two near-identical leaves would merge straight back.
    """
    depth, order = node_depths(graph), graph.topological_order()
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if (depth[a] != depth[b] or graph.nodes[a].kind != graph.nodes[b].kind
                    or set(graph.parents(a)) & set(graph.parents(b))
                    or b in graph.descendants(a) or a in graph.descendants(b)):
                continue
            pairs.append((a, b))
    return pairs


def merge_candidates(model, loader, device, limit_batches=None) -> list[tuple[float, str, str]]:
    """[(symmetric KL, keep, drop)] for every candidate pair, closest first.

    KL is between the two nodes' sub-DAG class distributions, averaged over samples
    (eval mode, no gradients).
    """
    pairs = candidate_pairs(model.graph)
    if not pairs:
        return []
    was_training = model.training
    model.eval()
    sums, total = {p: 0.0 for p in pairs}, 0
    with torch.no_grad():
        for x, _ in islice(loader, limit_batches):
            outs = node_outputs(model, model.route(x.to(device)))
            for a, b in pairs:
                la, lb = outs[a], outs[b]
                skl = ((la.exp() - lb.exp()) * (la - lb)).sum(1)  # KL(a||b) + KL(b||a)
                sums[(a, b)] += skl.sum().item()
            total += x.shape[0]
    model.train(was_training)
    return sorted((s / max(total, 1), a, b) for (a, b), s in sums.items())


def merge_nodes(model, keep: str, drop: str) -> list[str]:
    """Merge node `drop` into `keep` in a DAGRouterModel, in place (see RouterGraph.merge).

    Removes the modules of `drop` and of every node left unreachable; returns their names.
    Routers keep their output width because redirected edges keep their column.
    """
    removed = model.graph.merge(keep, drop)
    for n in removed:
        del model.nodes[n]
    return removed


def merge_step(model, loader, device, epoch: int, run_dir, threshold: float,
               limit_batches=None) -> list[dict]:
    """Merge growth step: repeatedly merge the closest candidate pair while its
    symmetric KL is below `threshold`, re-measuring after each merge.

    Each merge appends an event to <run_dir>/growth.jsonl; if nothing merges, one
    `"merge": null` line records the threshold and every candidate's KL instead.
    Writes a graph snapshot after any merge. Returns the merge events.
    """
    events = []
    while True:
        cands = merge_candidates(model, loader, device, limit_batches)
        if not cands or cands[0][0] >= threshold:
            break
        kl, keep, drop = cands[0]
        removed = merge_nodes(model, keep, drop)
        events.append({"epoch": epoch, "merge": [keep, drop], "kl": kl, "removed": removed,
                       "num_nodes": len(model.graph.nodes), "threshold": threshold})
        _log_event(run_dir, events[-1])
    if events:
        save_graph_snapshot(model, run_dir, epoch)
    else:
        _log_event(run_dir, {"epoch": epoch, "merge": None, "threshold": threshold,
                             "candidates": {f"{a}+{b}": kl for kl, a, b in cands}})
    return events


def _log_event(run_dir, event: dict) -> None:
    with open(Path(run_dir) / "growth.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")


def rebuild_optimizer(model, opt: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """New optimiser of the same type and hyperparameters over all current params.

    Params that survive the growth step keep their state (e.g. Adam moments);
    new params start with fresh state. The split leaf's old params are dropped.
    """
    defaults = {k: v for k, v in opt.param_groups[0].items() if k != "params"}
    new = type(opt)(model.parameters(), **defaults)
    for p in model.parameters():
        if p in opt.state:
            new.state[p] = opt.state[p]
    return new


def save_graph_snapshot(model, run_dir, epoch: int) -> Path:
    path = Path(run_dir) / "graphs" / f"epoch_{epoch:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.graph.to_dict(), indent=2))
    return path


def grow(model, loader, device, epoch: int, run_dir, max_nodes: int,
         limit_batches=None, noise: float = 1e-3) -> dict | None:
    """One growth step: split the leaf with the highest reach-weighted loss.

    Does nothing (returns None) if a split would exceed `max_nodes` graph nodes.
    Otherwise appends the event to <run_dir>/growth.jsonl, writes a graph
    snapshot to <run_dir>/graphs/epoch_NNN.json, and returns the event.
    """
    if len(model.graph.nodes) + 2 > max_nodes:
        return None
    losses = leaf_losses(model, loader, device, limit_batches)
    target = choose_leaf(losses)
    new = split_leaf(model, target, noise)
    event = {"epoch": epoch, "split": target, "new_nodes": list(new),
             "num_nodes": len(model.graph.nodes), "leaf_losses": losses}
    with open(Path(run_dir) / "growth.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")
    save_graph_snapshot(model, run_dir, epoch)
    return event
