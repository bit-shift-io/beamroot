"""Growth steps: split the worst leaf of a DAG router model (see CONTEXT.md: Growth step, Split)."""
import copy
import json
from itertools import islice
from pathlib import Path

import torch

from beamroot.losses import leaf_path_terms


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
            terms = leaf_path_terms(out["log_reach"], out["leaf_log_probs"], y)
            for name, t in terms.items():
                sums[name] = sums.get(name, 0.0) + t.sum().item()
            total += len(y)
    model.train(was_training)
    return {n: s / max(total, 1) for n, s in sums.items()}


def choose_leaf(losses: dict[str, float]) -> str:
    """The leaf with the highest reach-weighted loss."""
    return max(losses, key=losses.get)


def split_leaf(model, name: str, noise: float = 1e-3) -> tuple[str, str]:
    """Copy split of leaf `name` of a DAGRouterModel, in place: `cluster_split`'s
    fallback when the leaf's images do not form two groups.

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


def two_means(h: torch.Tensor, w: torch.Tensor, iters: int = 25) -> torch.Tensor:
    """Reach-weighted 2-means on rows of `h` (N, D) with weights `w` (N,); labels (N,) in {0, 1}.

    Deterministic start: the heaviest point, then the point maximising weight * squared
    distance from it. Ties in assignment go to cluster 0, so identical rows all land there.
    """
    c0 = h[w.argmax()]
    c1 = h[(w * ((h - c0) ** 2).sum(1)).argmax()]
    centres = torch.stack([c0, c1])
    for _ in range(iters):
        labels = torch.cdist(h, centres).argmin(1)
        for k in range(2):
            mask = (labels == k).float() * w
            if mask.sum() > 0:
                centres[k] = (mask[:, None] * h).sum(0) / mask.sum()
    return torch.cdist(h, centres).argmin(1)


def neuron_masks(h: torch.Tensor, w: torch.Tensor,
                 labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Which hidden units each group keeps: the group whose reach-weighted mean
    activation is higher. Every unit goes to exactly one group; ties go to group 0."""
    means = []
    for k in range(2):
        wk = (labels == k).float() * w
        means.append((wk[:, None] * h).sum(0) / wk.sum().clamp(min=1e-12))
    keep0 = means[0] >= means[1]
    return keep0, ~keep0


def _leaf_inputs(model, name, loader, device, limit_batches):
    """(stem features, hidden activations of leaf `name`, reach of `name`) over the
    loader's batches, in eval mode (no dropout) and without gradients."""
    was_training = model.training
    model.eval()
    feats, hidden, reach = [], [], []
    with torch.no_grad():
        for x, _ in islice(loader, limit_batches):
            x = x.to(device)
            f = model.stem(x).flatten(1)
            feats.append(f)
            hidden.append(model.nodes[name].net[:2](f))  # output of the ReLU
            reach.append(model.route(x)["log_reach"][name].exp())
    model.train(was_training)
    return torch.cat(feats), torch.cat(hidden), torch.cat(reach)


def _masked_leaf(old, fresh, keep: torch.Tensor, noise: float):
    """Copy of `old` keeping hidden units `keep` (plus noise); the other units are
    taken from the freshly initialised leaf `fresh`."""
    leaf = copy.deepcopy(old)
    first, last = leaf.net[0], leaf.net[3]
    with torch.no_grad():
        for p in leaf.parameters():
            p.add_(torch.randn_like(p) * noise)
        drop = ~keep
        first.weight[drop] = fresh.net[0].weight[drop]
        first.bias[drop] = fresh.net[0].bias[drop]
        last.weight[:, drop] = fresh.net[3].weight[:, drop]
    return leaf


def _pretrain_router(router, feats, labels, w, steps: int, lr: float) -> float:
    """Fit `router` to the group labels with reach-weighted cross-entropy, using a
    throwaway Adam. Returns the final weighted loss."""
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    w = w / w.sum().clamp(min=1e-12)

    def loss_fn():
        return -(w * router(feats).gather(1, labels[:, None]).squeeze(1)).sum()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return loss_fn().item()


def cluster_split(model, name: str, loader, device, noise: float = 1e-3,
                  limit_batches=None, pretrain_steps: int = 100,
                  pretrain_lr: float = 1e-2) -> tuple[tuple[str, str], dict]:
    """Split leaf `name` with cluster initialisation (see CONTEXT.md: Split initialisation).

    The leaf's images (the loader's first `limit_batches` batches) are split in two by
    reach-weighted 2-means on its hidden activations. New leaf i copies the old leaf,
    keeps (plus `noise`) the hidden units group i activates more, and re-initialises
    the rest. The new router is pre-trained on the group labels. If one group is
    empty, falls back to a `copy` split. Returns (new leaves, info for growth.jsonl).
    """
    feats, hidden, reach = _leaf_inputs(model, name, loader, device, limit_batches)
    labels = two_means(hidden, reach)
    sizes = [int((labels == k).sum()) for k in range(2)]
    info = {"cluster_sizes": sizes,
            "cluster_reach": [float(reach[labels == k].sum()) for k in range(2)],
            "router_pretrain_loss": None, "fallback": None}
    if min(sizes) == 0:
        info["fallback"] = "copy"
        return split_leaf(model, name, noise), info
    old = model.nodes[name]
    masks = neuron_masks(hidden, reach, labels)
    new = model.graph.split(name)
    model.nodes[name] = model.build_node(name).to(feats.device)
    for n, keep in zip(new, masks):
        fresh = model.build_node(n).to(feats.device)
        model.nodes[n] = _masked_leaf(old, fresh, keep, noise)
    info["router_pretrain_loss"] = _pretrain_router(
        model.nodes[name], feats, labels, reach, pretrain_steps, pretrain_lr)
    return new, info


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


def candidate_pairs(graph, protected=()) -> list[tuple[str, str]]:
    """(keep, drop) pairs that may merge: same kind, no shared parent, neither an ancestor
    of the other; depth does not matter. `keep` is the earlier one topologically.

    Siblings are excluded: merging them would need one router output column to vanish,
    and every fresh split's two near-identical leaves would merge straight back.
    Ancestor/descendant pairs are excluded so a merge can never form a loop (ADR 0002).
    Nodes in `protected` (the latest split's new leaves) are never candidates.
    """
    order = [n for n in graph.topological_order() if n not in protected]
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if (graph.nodes[a].kind != graph.nodes[b].kind
                    or set(graph.parents(a)) & set(graph.parents(b))
                    or b in graph.descendants(a) or a in graph.descendants(b)):
                continue
            pairs.append((a, b))
    return pairs


def merge_candidates(model, loader, device, limit_batches=None,
                     protected=()) -> list[tuple[float, str, str]]:
    """[(symmetric KL, keep, drop)] for every candidate pair (see `candidate_pairs`), closest first.

    KL is between the two nodes' sub-DAG class distributions, averaged over samples
    (eval mode, no gradients).
    """
    pairs = candidate_pairs(model.graph, protected)
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
               limit_batches=None, protected=()) -> list[dict]:
    """Merge growth step: repeatedly merge the closest candidate pair while its
    symmetric KL is below `threshold`, re-measuring after each merge. Nodes in
    `protected` (the previous growth step's new leaves) are never candidates.

    Each merge appends an event to <run_dir>/growth.jsonl; if nothing merges, one
    `"merge": null` line records the threshold and every candidate's KL instead.
    Writes a graph snapshot after any merge. Returns the merge events.
    """
    events = []
    while True:
        cands = merge_candidates(model, loader, device, limit_batches, protected)
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
    """One growth step: cluster-split the leaf with the highest reach-weighted loss
    (ADR 0003), clustering the same batches the loss is measured on.

    Does nothing (returns None) if a split would exceed `max_nodes` graph nodes.
    Otherwise appends the event (with cluster sizes, router pre-train loss and any
    fallback) to <run_dir>/growth.jsonl, writes a graph snapshot to
    <run_dir>/graphs/epoch_NNN.json, and returns the event.
    """
    if len(model.graph.nodes) + 2 > max_nodes:
        return None
    losses = leaf_losses(model, loader, device, limit_batches)
    target = choose_leaf(losses)
    new, info = cluster_split(model, target, loader, device, noise, limit_batches)
    event = {"epoch": epoch, "split": target, "new_nodes": list(new),
             "num_nodes": len(model.graph.nodes), "leaf_losses": losses, **info}
    with open(Path(run_dir) / "growth.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")
    save_graph_snapshot(model, run_dir, epoch)
    return event
