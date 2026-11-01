"""Growth steps: split the worst leaf of a DAG router model (see CONTEXT.md: Growth step, Split)."""
import copy
import json
import math
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from beamroot.cost import model_costs
from beamroot.losses import leaf_path_terms
from beamroot.models.tree import BLOCKS, CONVS, RouterHead, TreeModel, TreeNode


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


def _set_steps(model, name: str, new) -> None:
    """Cascade only: new nodes take the split leaf's size step (before `build_node`)."""
    if hasattr(model, "steps"):
        for n in new:
            model.steps[n] = model.steps[name]


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
    _set_steps(model, name, new)
    model.nodes[name] = model.build_node(name).to(ref.device)
    for n in new:
        if hasattr(model, "steps"):  # cascade: random leaves at the parent's step, no copy
            model.nodes[n] = model.build_node(n).to(ref.device)
            continue
        leaf = copy.deepcopy(old)
        with torch.no_grad():
            for p in leaf.parameters():
                p.add_(torch.randn_like(p) * noise)
        model.nodes[n] = leaf
    return new


def _narrow(child: TreeNode, keep: list[torch.Tensor]) -> None:
    """Narrow `child`'s copied segment in place to the kept output channels of each conv
    (`keep`, one index tensor per conv in segment order). Each conv's input follows the
    previous conv's kept channels (the first conv reads its parent's output unchanged), and
    the classifier's input follows the last. Kept weights are copied exactly."""
    convs = [(blk, i) for blk in child.segment for i, m in enumerate(blk)
             if isinstance(m, nn.Conv2d)]
    if len(keep) != len(convs):
        raise ValueError(f"keep has {len(keep)} layers, the branch has {len(convs)} convs")
    prev = None
    for (blk, i), out in zip(convs, keep):
        old = blk[i]
        out = out.to(old.weight.device)
        w = old.weight[out] if prev is None else old.weight[out][:, prev]
        new = nn.utils.skip_init(nn.Conv2d, w.shape[1], w.shape[0], old.kernel_size,
                                 padding=old.padding, device=w.device, dtype=w.dtype)
        with torch.no_grad():
            new.weight.copy_(w)
            new.bias.copy_(old.bias[out])
        blk[i], prev = new, out
    if prev is not None:
        old = child.head.linear
        new = nn.utils.skip_init(nn.Linear, len(prev), old.out_features,
                                 device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(old.weight[:, prev])
            new.bias.copy_(old.bias)
        child.head.linear = new


def tree_split(model, leaf: str, groups: list[list[int]], block: str,
               keep: list[list[torch.Tensor]] | None = None) -> tuple[str, ...]:
    """Split leaf `leaf` of a `TreeModel` at split block `block` (a `tree.BLOCKS` name),
    in place. Returns the new leaves' names, in the order of `groups`.

    The leaf's segment up to and including `block` stays with it, now a router whose fresh
    `RouterHead` reads that block's output and routes to `groups` (which must partition
    the leaf's classes). Each child leaf is a copy of the remaining blocks and the
    classifier. With `keep` None the copy is exact and full width, so every image's leaf
    output is unchanged. Otherwise `keep[i]` narrows child i (CONTEXT.md: Branch keep): one
    tensor of kept output channel indices per conv of the child's segment, in order (empty
    if the child owns no blocks).
    """
    if model.graph.nodes[leaf].kind != "leaf":
        raise ValueError(f"only leaves can be split, {leaf!r} is a router")
    start = model.start_block(leaf)
    if block not in BLOCKS or BLOCKS.index(block) < start:
        raise ValueError(f"block {block!r} is not in leaf {leaf!r}'s segment")
    if sorted(c for g in groups for c in g) != sorted(model.node_classes(leaf)) \
            or any(not g for g in groups):
        raise ValueError(f"groups {groups} do not partition leaf {leaf!r}'s classes")
    old = model.nodes[leaf]
    ref = next(old.parameters())
    cut = BLOCKS.index(block) + 1 - start  # blocks of the old segment the router keeps
    widths = model.node_widths(leaf)
    n_convs = sum(CONVS[start:start + cut])  # convs the router keeps
    new = model.graph.split(leaf, len(groups))
    model.split_blocks[leaf] = block
    model.groups[leaf] = [list(g) for g in groups]
    model.set_widths(leaf, widths[:n_convs])
    for i, n in enumerate(new):
        model.set_widths(n, widths[n_convs:] if keep is None else [len(k) for k in keep[i]])
    router = model.build_node(leaf).to(ref.device)
    blocks = list(old.segment)  # fresh Sequentials renumber keys, matching `build_node`
    router.segment = nn.Sequential(*blocks[:cut])
    model.nodes[leaf] = router
    for i, n in enumerate(new):
        model.nodes[n] = TreeNode(copy.deepcopy(nn.Sequential(*blocks[cut:])),
                                  copy.deepcopy(old.head))
        if keep is not None:
            _narrow(model.nodes[n], keep[i])
    return new


def confusion_groups(confusion: torch.Tensor, classes: list[int]) -> list[list[int]]:
    """Confusion groups (CONTEXT.md) of a leaf: the top cut of average-linkage clustering
    on 1 - the symmetric confusion rate (experiment 0007's `hierarchy()`).

    `confusion[i, j]` counts validation images of class `classes[i]` the leaf predicted as
    `classes[j]`. Returns two groups of class ids, each sorted, ordered by smallest class.
    """
    conf = confusion.detach().float().cpu()
    rate = conf / conf.sum(1, keepdim=True).clamp(min=1)
    sim = (rate + rate.T) / 2
    clusters = [[i] for i in range(len(classes))]
    while len(clusters) > 2:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = float(sim[clusters[i]][:, clusters[j]].mean())
                if best is None or s > best[0]:
                    best = (s, i, j)
        _, i, j = best
        clusters[i] = clusters[i] + clusters.pop(j)
    return sorted(sorted(classes[k] for k in c) for c in clusters)


def choose_split_block(block_val_acc: dict[str, float], route_acc_min: float
                       ) -> tuple[str, str | None]:
    """Split block from per-block validation route accuracy (in block order): the earliest
    block at or above `route_acc_min`, else the last block with fallback "last_block".
    Returns (block, fallback or None)."""
    for block, acc in block_val_acc.items():
        if acc >= route_acc_min:
            return block, None
    return list(block_val_acc)[-1], "last_block"


def choose_root_block(block_val_acc: dict[str, float], route_acc_min: float) -> str | None:
    """Split block for the root (DECISIONS Q5): the earliest block at or above
    `route_acc_min`, excluding the last block, whose children would own no blocks. None
    means no block qualifies: skip this growth step and probe again at the next."""
    for block, acc in list(block_val_acc.items())[:-1]:
        if acc >= route_acc_min:
            return block
    return None


def keep_channels(mass: torch.Tensor, share: float) -> torch.Tensor:
    """Branch keep for one conv layer (CONTEXT.md): the indices, ascending, of the fewest
    channels whose activation mass sums to at least `share` of the layer's total. Channels
    with no mass are never kept below share 1.0; share 1.0 keeps every channel; at least one
    channel is always kept."""
    mass = mass.detach().float().cpu()
    if share >= 1.0:
        return torch.arange(len(mass))
    order = torch.sort(mass, descending=True, stable=True).indices
    reached = torch.cumsum(mass[order], 0) >= share * mass.sum()
    count = int(reached.long().argmax()) + 1 if mass.sum() > 0 else 1
    return order[:count].sort().values


PROBE_EPOCHS = 8  # probe head epochs over the frozen growth-batch features
PROBE_BATCH = 128  # probe minibatch; also bounds memory on MPS
PROBE_LR = 1e-3


@contextmanager
def _rng_untouched(loaders):
    """Restore the global RNG (CPU and MPS) and the loaders' shuffle generators on exit, so
    a growth step draws nothing from the training RNG (data augmentation, init, shuffles)."""
    cpu = torch.get_rng_state()
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    gens = [(g, g.get_state()) for g in (getattr(ld, "generator", None) for ld in loaders)
            if g is not None]
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if mps is not None:
            torch.mps.set_rng_state(mps)
        for g, state in gens:
            g.set_state(state)


def _true_path_input(model, leaf: str, x: torch.Tensor, y: torch.Tensor):
    """`leaf`'s input for the images of batch `x` whose true path reaches it, and their
    labels: each router on the path keeps only the rows of the child's group."""
    path = [leaf]
    while model.graph.parents(path[-1]):
        path.append(model.graph.parents(path[-1])[0])
    path.reverse()
    for node, child in zip(path, path[1:]):
        keep = torch.zeros(model.num_classes, dtype=torch.bool, device=y.device)
        keep[model.groups[node][model.graph.children(node).index(child)]] = True
        rows = keep[y]
        x, y = model.nodes[node].segment(x[rows]), y[rows]
    return x, y


def _leaf_block_feats(model, leaf, probes, loader, device, limit_batches, groups=None):
    """Pooled outputs of each block of `leaf`'s segment (each probe's pool), the leaf's
    logits and the labels, over the images of `loader` whose true path reaches `leaf`.

    With `groups`, also returns each group's activation mass per conv of the segment
    (`mass[g][j]`: per channel, the sum of conv j's post-ReLU output over group g's images
    and spatial positions), from the same batches."""
    feats = {b: [] for b in probes}
    logits, labels = [], []
    mass = [[] for _ in groups or []]
    for x, y in islice(loader, limit_batches):
        h, y = _true_path_input(model, leaf, x.to(device), y.to(device))
        g = _group_labels(groups, y, model.num_classes) if groups else None
        node = model.nodes[leaf]
        conv = 0
        for block, layer in zip(probes, node.segment):
            for m in layer:
                h = m(h)
                if groups and isinstance(m, nn.ReLU):
                    for i, per_conv in enumerate(mass):
                        s = h[g == i].sum(dim=(0, 2, 3))
                        if conv < len(per_conv):
                            per_conv[conv] = per_conv[conv] + s
                        else:
                            per_conv.append(s)
                    conv += 1
            feats[block].append(probes[block].pool(h))
        logits.append(node.head(h))
        labels.append(y)
    out = ({b: torch.cat(f) for b, f in feats.items()}, torch.cat(logits), torch.cat(labels))
    return (*out, mass) if groups else out


def _fit_probe(head, feats, labels, gen: torch.Generator) -> None:
    """Train `head` (on pooled features) to the group labels with its own Adam and
    minibatch order from `gen`."""
    opt = torch.optim.Adam(head.net.parameters(), lr=PROBE_LR)
    for _ in range(PROBE_EPOCHS):
        order = torch.randperm(len(feats), generator=gen).to(feats.device)
        for s in range(0, len(feats), PROBE_BATCH):
            idx = order[s:s + PROBE_BATCH]
            opt.zero_grad()
            F.cross_entropy(head.net(feats[idx]), labels[idx]).backward()
            opt.step()


def _probe_acc(head, feats, labels) -> float:
    with torch.no_grad():
        hits = sum((head.net(feats[s:s + 512]).argmax(1) == labels[s:s + 512]).sum().item()
                   for s in range(0, len(feats), 512))
    return hits / max(len(labels), 1)


def _group_labels(groups, y, num_classes):
    lookup = torch.zeros(num_classes, dtype=torch.long, device=y.device)
    for i, g in enumerate(groups):
        lookup[g] = i
    return lookup[y]


def _grow_tree(model, loader, device, epoch, run_dir, max_nodes, limit_batches, val_loader,
               route_acc_min, branch_keep):
    """`grow` for a `TreeModel` (CONTEXT.md: Confusion groups, Split block, Route accuracy)."""
    if len(model.graph.nodes) + 2 > max_nodes:
        return None
    with _rng_untouched([loader, val_loader]):
        was_training = model.training
        model.eval()
        losses = leaf_losses(model, loader, device, limit_batches)
        eligible = {n: v for n, v in losses.items() if len(model.node_classes(n)) >= 2
                    and model.start_block(n) < len(BLOCKS)}
        if not eligible:
            model.train(was_training)
            return None
        target = choose_leaf(eligible)
        classes = model.node_classes(target)
        torch.manual_seed(0)  # probe init; the caller's RNG state is restored on exit
        cfg = model.config
        start = model.start_block(target)
        probes = {BLOCKS[b]: RouterHead(model.block_width(target, b),
                                        cfg["image_size"] // 2 ** (b + 1), 2).to(device)
                  for b in range(start, len(BLOCKS))}
        with torch.no_grad():
            val_feats, val_logits, val_y = _leaf_block_feats(model, target, probes, val_loader,
                                                             device, limit_batches)
        cls = torch.tensor(classes, device=val_y.device)
        pos = torch.zeros(model.num_classes, dtype=torch.long, device=val_y.device)
        pos[cls] = torch.arange(len(classes), device=val_y.device)
        pred = val_logits[:, cls].argmax(1)
        k = len(classes)
        confusion = torch.bincount((pos[val_y] * k + pred).cpu(), minlength=k * k).view(k, k)
        groups = confusion_groups(confusion, classes)
        with torch.no_grad():  # the probes' batches also measure each group's activation mass
            tr_feats, _, tr_y, mass = _leaf_block_feats(model, target, probes, loader, device,
                                                        limit_batches, groups)
        tr_g = _group_labels(groups, tr_y, model.num_classes)
        val_g = _group_labels(groups, val_y, model.num_classes)
        gen = torch.Generator().manual_seed(0)
        block_val_acc = {}
        for block, head in probes.items():
            _fit_probe(head, tr_feats[block], tr_g, gen)
            block_val_acc[block] = _probe_acc(head, val_feats[block], val_g)
        root = target == model.graph.root
        if root:
            block, fallback = choose_root_block(block_val_acc, route_acc_min), None
        else:
            block, fallback = choose_split_block(block_val_acc, route_acc_min)
        skip = block is None
        if not skip:
            n_convs = sum(CONVS[start:BLOCKS.index(block) + 1])  # convs the router keeps
            keep = None if branch_keep >= 1.0 else [
                [keep_channels(m, branch_keep) for m in per_conv[n_convs:]] for per_conv in mass]
            new = tree_split(model, target, groups, block, keep=keep)
            model.nodes[target].head.net.load_state_dict(probes[block].net.state_dict())
            node_macs = model_costs(model)[1]
        model.train(was_training)
    if skip:  # the root only splits at a block that routes well; retry next growth step
        event = {"epoch": epoch, "leaf": target, "skipped": "root_below_bar",
                 "num_nodes": len(model.graph.nodes), "leaf_losses": losses, "groups": groups,
                 "block_val_acc": block_val_acc, "excluded_blocks": [BLOCKS[-1]]}
        _log_event(run_dir, event)
        return event
    event = {"epoch": epoch, "split": target, "new_nodes": list(new),
             "num_nodes": len(model.graph.nodes), "leaf_losses": losses, "groups": groups,
             "block": block, "block_val_acc": block_val_acc, "fallback": fallback,
             "branch_keep": branch_keep,
             "branch_channels": {n: model.node_widths(n) for n in new},
             "branch_macs": {n: node_macs[n] for n in new}}
    if root:
        event["excluded_blocks"] = [BLOCKS[-1]]  # probed and logged, never chosen
    _log_event(run_dir, event)
    save_graph_snapshot(model, run_dir, epoch)
    return event


K_MARGIN = 0.05  # a larger k must beat k = 2's silhouette by this much to be picked
SILHOUETTE_MAX_POINTS = 8000  # silhouette is O(N^2); evenly subsample beyond this


def k_means(h: torch.Tensor, w: torch.Tensor, k: int = 2, iters: int = 25) -> torch.Tensor:
    """Reach-weighted k-means on rows of `h` (N, D) with weights `w` (N,); labels (N,) in 0..k-1.

    Deterministic start (farthest-point picks): the heaviest point, then repeatedly the
    point maximising weight * squared distance to its nearest chosen centre. Ties in
    assignment go to the lowest cluster, so identical rows all land in cluster 0.
    """
    centres = [h[w.argmax()]]
    for _ in range(k - 1):
        d2 = torch.cdist(h, torch.stack(centres)).pow(2).amin(1)
        centres.append(h[(w * d2).argmax()])
    centres = torch.stack(centres)
    for _ in range(iters):
        labels = torch.cdist(h, centres).argmin(1)
        for c in range(k):
            mask = (labels == c).float() * w
            if mask.sum() > 0:
                centres[c] = (mask[:, None] * h).sum(0) / mask.sum()
    return torch.cdist(h, centres).argmin(1)


def silhouette(h: torch.Tensor, w: torch.Tensor, labels: torch.Tensor) -> float:
    """Reach-weighted mean silhouette of a clustering (in [-1, 1]; higher is better separated).

    Point i's a is its weighted mean distance to the rest of its own cluster, b the
    lowest weighted mean distance to another non-empty cluster; s = (b - a) / max(a, b),
    0 for a point alone in its cluster. Computed in float32.
    """
    if len(h) > SILHOUETTE_MAX_POINTS:
        idx = torch.linspace(0, len(h) - 1, SILHOUETTE_MAX_POINTS).long().to(h.device)
        h, w, labels = h[idx], w[idx], labels[idx]
    h, w = h.float(), w.float()
    k = int(labels.max()) + 1
    onehot = torch.nn.functional.one_hot(labels, k).float() * w[:, None]  # (N, k)
    sums = torch.cdist(h, h) @ onehot  # weighted distance sums to each cluster
    mass = onehot.sum(0)  # (k,)
    own = mass[labels] - w
    a = sums.gather(1, labels[:, None]).squeeze(1) / own.clamp(min=1e-12)
    other = sums / mass.clamp(min=1e-12)
    other[torch.arange(len(h)), labels] = float("inf")
    other[:, mass == 0] = float("inf")
    b = other.amin(1)
    s = (b - a) / torch.maximum(a, b).clamp(min=1e-12)
    s = torch.where(own > 1e-12, s, torch.zeros_like(s))
    return float((w * s).sum() / w.sum().clamp(min=1e-12))


def pick_k(h: torch.Tensor, w: torch.Tensor, max_k: int = 2):
    """Cluster for k = 2..max_k and pick k by reach-weighted silhouette.

    Keeps k = 2 unless a larger k with no empty group beats its silhouette by more
    than `K_MARGIN`. Returns (k, labels for that k, {k: silhouette}).
    """
    labels, scores = {}, {}
    for k in range(2, max_k + 1):
        labels[k] = k_means(h, w, k)
        scores[k] = silhouette(h, w, labels[k])
    full = [k for k in labels if len(set(labels[k].tolist())) == k]
    best = max(full, key=scores.get, default=2)
    k = best if best != 2 and scores[best] > scores[2] + K_MARGIN else 2
    return k, labels[k], scores


def neuron_masks(h: torch.Tensor, w: torch.Tensor, labels: torch.Tensor,
                 k: int = 2) -> list[torch.Tensor]:
    """Which hidden units each of the `k` groups keeps: the group whose reach-weighted
    mean activation is highest. Every unit goes to exactly one group; ties go to the
    lowest group."""
    means = []
    for c in range(k):
        wc = (labels == c).float() * w
        means.append((wc[:, None] * h).sum(0) / wc.sum().clamp(min=1e-12))
    best, top = torch.zeros_like(means[0], dtype=torch.long), means[0]
    for c in range(1, k):
        wins = means[c] > top
        best, top = torch.where(wins, c, best), torch.where(wins, means[c], top)
    return [best == c for c in range(k)]


CLASS_MIN_SHARE = 0.1  # class split policy: a class needs this share of the leaf's reach


def class_groups(y: torch.Tensor, w: torch.Tensor, max_k: int = 2):
    """Group a leaf's images by label for the class split policy.

    Each class with at least `CLASS_MIN_SHARE` of the reach-weighted mass `w` gets its
    own group; the other classes share one. With more than `max_k` groups the smallest
    qualifying classes are folded into the shared group. Returns
    (classes per group, group index per image), qualifying classes first in class order
    and the shared group last; or None if no class qualifies or fewer than two groups
    would result (the caller falls back to the cluster policy).
    """
    present = sorted(set(y.tolist()))
    total = w.sum().clamp(min=1e-12)
    mass = {c: float(w[y == c].sum() / total) for c in present}
    own = [c for c in present if mass[c] >= CLASS_MIN_SHARE]
    if not own:
        return None
    if len(own) + (len(own) < len(present)) > max_k:
        own = sorted(sorted(own, key=lambda c: -mass[c])[:max_k - 1])
    rest = [c for c in present if c not in own]
    groups = [[c] for c in own] + ([rest] if rest else [])
    if len(groups) < 2:
        return None
    labels = torch.empty_like(y)
    for i, g in enumerate(groups):
        for c in g:  # no torch.isin: not implemented on MPS
            labels[y == c] = i
    return groups, labels


LINK_RATIO = 0.5  # loss link policy: a leaf links if its loss is below this x the split leaf's


def choose_links(group_losses: list[dict[str, float]],
                 own: list[float]) -> dict[int, tuple[str, float]]:
    """Which split groups link to an existing leaf (loss link policy, see CONTEXT.md: Link).

    `group_losses[g]` maps each candidate leaf to its mean NLL on group g's validation
    images, `own[g]` is the split leaf's. A group links to its lowest-loss leaf if that
    loss is below `LINK_RATIO` x `own[g]`; groups with `own[g]` of 0 never link. If
    every group would link, the weakest link (highest ratio) stays fresh.
    Returns {group: (leaf, ratio)}.
    """
    links = {}
    for g, (losses, base) in enumerate(zip(group_losses, own)):
        if not losses or base <= 0:
            continue
        leaf = min(losses, key=losses.get)
        ratio = losses[leaf] / base
        if ratio < LINK_RATIO:
            links[g] = (leaf, ratio)
    if links and len(links) == len(own):
        del links[max(links, key=lambda g: links[g][1])]
    return links


LINK_PURITY = 0.9  # class link policy: a leaf links if its class purity is at least this


def choose_class_links(group_classes: list, leaf_stats: dict[str, tuple]) -> dict[int, tuple]:
    """Which split groups link to an existing leaf (class link policy, see CONTEXT.md: Link).

    `group_classes[g]` is group g's dominant class on the validation set (None if empty),
    `leaf_stats` maps each candidate leaf to (dominant class, purity). A group links to
    the highest-purity leaf of the same class whose purity is >= `LINK_PURITY`. If every
    group would link, the weakest link (lowest purity) stays fresh.
    Returns {group: (leaf, class, purity)}.
    """
    links = {}
    for g, cls in enumerate(group_classes):
        cands = [(pur, leaf) for leaf, (c, pur) in leaf_stats.items()
                 if cls is not None and c == cls and pur >= LINK_PURITY]
        if cands:
            pur, leaf = max(cands, key=lambda t: t[0])
            links[g] = (leaf, cls, pur)
    if links and len(links) == len(group_classes):
        del links[min(links, key=lambda g: links[g][2])]
    return links


def _val_class_mass(model, name, loader, device, limit_batches, assign, k):
    """Validation class mass on the existing graph (eval mode, no gradients):
    ([k group masses], {other leaf: mass}), each a per-class list. A group's mass is
    reach of `name` summed per class over the images `assign` puts in it; a leaf's is
    its own reach per class."""
    leaves = [n for n in model.graph.leaves() if n != name]
    was_training = model.training
    model.eval()
    gm, lm = None, None
    with torch.no_grad():
        for x, y in islice(loader, limit_batches):
            x, y = x.to(device), y.to(device)
            out = model.route(x)
            c = out["leaf_log_probs"][name].shape[1]
            if gm is None:
                gm = torch.zeros(k, c, device=device)
                lm = {n: torch.zeros(c, device=device) for n in leaves}
            g = assign(model.nodes[name].hidden(model.features(x)), y)
            r = out["log_reach"][name].exp()
            for i in range(k):
                gm[i].index_add_(0, y[g == i], r[g == i])
            for n in leaves:
                lm[n].index_add_(0, y, out["log_reach"][n].exp())
    model.train(was_training)
    return gm.tolist(), {n: v.tolist() for n, v in lm.items()}


def _view_parent(model, name, out) -> str:
    """The parent router of label leaf `name` sending it the most reach (on this batch)."""
    def sent(p):
        i = model.graph.children(p).index(name)
        return (out["log_reach"][p] + out["log_route"][p][:, i]).exp().sum().item()
    return max(model.graph.parents(name), key=sent)


def _leaf_inputs(model, name, loader, device, limit_batches):
    """(stem features, hidden activations of leaf `name`, reach of `name`, labels) over
    the loader's batches, in eval mode (no dropout) and without gradients."""
    was_training = model.training
    model.eval()
    feats, hidden, reach, ys = [], [], [], []
    parent = None
    with torch.no_grad():
        for x, y in islice(loader, limit_batches):
            ys.append(y.to(device))
            x = x.to(device)
            f = model.features(x)
            feats.append(f)
            out = model.route(x)
            if hasattr(model.nodes[name], "hidden"):
                hidden.append(model.nodes[name].hidden(f))  # output of the ReLU
            else:  # label leaf: the highest-reach parent router's view
                if parent is None:
                    parent = _view_parent(model, name, out)
                hidden.append(model.nodes[parent].hidden(f))
            reach.append(out["log_reach"][name].exp())
    model.train(was_training)
    return torch.cat(feats), torch.cat(hidden), torch.cat(reach), torch.cat(ys)


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


def _pretrain_router(router, feats, labels, w, steps: int, lr: float,
                     chunk: int = 512) -> float:
    """Fit `router` to the group labels with reach-weighted cross-entropy, using a
    throwaway Adam. Each step is one full-batch update, accumulated over chunks so a big
    cascade router fits in memory. Returns the final weighted loss."""
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    w = w / w.sum().clamp(min=1e-12)

    def loss_fn(backward: bool) -> float:
        total = 0.0
        for s in range(0, len(feats), chunk):
            e = s + chunk
            loss = -(w[s:e] * router(feats[s:e]).gather(1, labels[s:e, None]).squeeze(1)).sum()
            if backward:
                loss.backward()
            total += loss.item()
        return total
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(backward=True)
        opt.step()
    with torch.no_grad():
        return loss_fn(backward=False)


MAX_STEP = 4


def _size_check(model, name: str, scores: dict, sizes: list, minimum: float,
                purity=None) -> dict:
    """Cascade size check (i): a poorly separated split (best silhouette < `minimum`, or an
    empty group) bumps the router and its new children one step above the split leaf.
    `purity` = (gain, minimum) optionally adds the purity check: it fails when the gain is
    <= the minimum. Either failure gives one bump; the record lists every check that ran
    (`check` is the first failing one, a purity check is also logged as `purity`).
    Sets `model.steps[name]` so the router is built at the bumped step; returns the event record.
    """
    if not scores and purity is None:
        return {"check": "skipped"}
    rec = {}
    sil_fail = False
    if scores:
        rec = {"check": "silhouette", "value": max(scores.values())}
        sil_fail = rec["value"] < minimum or min(sizes) == 0
    pur_fail = False
    if purity is not None:
        gain, pmin = purity
        pur_fail = gain <= pmin
        entry = {"check": "purity", "value": gain}
        if rec:
            rec["purity"] = entry
        else:
            rec = dict(entry)
    if not (sil_fail or pur_fail):
        return {**rec, "bumped": False}
    if pur_fail and not sil_fail and scores:
        rec["check"] = "purity"
        rec["value"], rec["silhouette"] = rec["purity"]["value"], rec["value"]
    s = model.steps[name]
    if s >= MAX_STEP:
        return {**rec, "at_max": True}
    model.steps[name] = s + 1
    return {**rec, "from": s, "to": s + 1}


def _purity_gain(ys, reach, labels, k, num_classes) -> float:
    """Reach-weighted purity of the k groups minus the leaf's own purity (class mass)."""
    from beamroot.analysis import purity
    mass = torch.zeros(k, num_classes, dtype=torch.float64)
    for g in range(k):
        sel = labels == g
        mass[g].index_add_(0, ys[sel].cpu(), reach[sel].cpu().double())
    total = mass.sum()
    if total <= 0:
        return 0.0
    leaf = purity(mass.sum(0).tolist())[1]
    groups = sum(purity(m.tolist())[1] * float(m.sum()) for m in mass) / float(total)
    return groups - leaf


def size_checks(model, stats: dict, entropy_max: float, spread_min: float, device,
                epoch: int = 0, cooldown: int = 0) -> list[dict]:
    """Cascade size checks (ii) and (iv) on last epoch's router `stats` ({router: usage, entropy}).

    A router whose entropy exceeds `entropy_max` * ln k, or whose smallest child usage is below
    `spread_min` * 2 / k, is rebuilt fresh one step up (same children and balance target).
    At `MAX_STEP` it is left alone. A router younger than `cooldown` epochs (since it was
    created or last bumped, `model.last_change_epoch`) is skipped, as is a frozen router
    (`model.frozen`). Returns the bump records; the caller rebuilds the optimiser.
    """
    records = []
    for name, s in stats.items():
        if name not in model.nodes or model.graph.nodes[name].kind != "router":
            continue
        if name in model.frozen:
            continue
        if cooldown and epoch - model.last_change_epoch.get(name, 0) < cooldown:
            continue
        k = len(s["usage"])
        if s["entropy"] > entropy_max * math.log(k):
            check, value = "entropy", s["entropy"]
        elif min(s["usage"]) < spread_min * 2 / k:
            check, value = "spread", min(s["usage"])
        else:
            continue
        step = model.steps[name]
        if step >= MAX_STEP:
            records.append({"bump": name, "check": check, "at_max": True, "value": value})
            continue
        model.steps[name] = step + 1
        ref = next(model.nodes[name].parameters())
        model.nodes[name] = model.build_node(name).to(ref.device)
        model.last_change_epoch[name] = epoch
        model.decay_start.pop(name, None)  # a bumped router is not an ancestor until the next split
        model.frozen.discard(name)  # fresh weights train (build_node gives requires_grad=True)
        records.append({"bump": name, "check": check, "from": step, "to": step + 1,
                        "value": value})
    return records


def freeze_checks(model, stats: dict, entropy_max: float, spread_min: float) -> list[dict]:
    """Router freeze on last epoch's router `stats` ({router: usage, entropy}).

    An unfrozen ancestor router (one in `model.decay_start`) whose entropy is at most
    `entropy_max` * ln k and whose smallest child usage is at least `spread_min` * 2 / k
    stops training: its params get requires_grad=False and it joins `model.frozen`.
    Returns one `{"freeze", "entropy", "spread"}` record per frozen router.
    """
    records = []
    for name, s in stats.items():
        if name not in model.decay_start or name in model.frozen or name not in model.nodes:
            continue
        k = len(s["usage"])
        spread = min(s["usage"])
        if s["entropy"] <= entropy_max * math.log(k) and spread >= spread_min * 2 / k:
            model.nodes[name].requires_grad_(False)
            model.frozen.add(name)
            records.append({"freeze": name, "entropy": s["entropy"], "spread": spread})
    return records


def cluster_split(model, name: str, loader, device, noise: float = 1e-3,
                  limit_batches=None, pretrain_steps: int = 100,
                  pretrain_lr: float = 1e-2, max_k: int = 2,
                  policy: str = "cluster", link_policy: str = "none",
                  val_loader=None, silhouette_min=None,
                  purity_gain=None, leaf_init: str = "fresh") -> tuple[tuple[str, ...], dict]:
    """Split leaf `name` with cluster initialisation (see CONTEXT.md: Split initialisation).

    The leaf's images (the loader's first `limit_batches` batches) are split into k
    groups by reach-weighted k-means on its hidden activations, k in 2..`max_k` picked
    by silhouette (see `pick_k`). New leaf i copies the old leaf, keeps (plus `noise`)
    the hidden units group i activates most, and re-initialises the rest. The new
    k-output router is pre-trained on the group labels, and its balance target is the
    groups' reach shares. If a group is empty, falls back to a 2-way `copy` split
    with no target (uniform). Returns (new leaves, info for growth.jsonl).

    With `policy="class"` the groups come from `class_groups` instead of k-means
    (k = the number of groups, info gains `groups`); if no class qualifies it falls
    back to the cluster policy and records `policy_fallback`.

    With `link_policy="class"`, a group links to an existing leaf of the same dominant
    validation class with purity >= `LINK_PURITY` (`choose_class_links`); info gains
    `links` {group: {"policy", "leaf", "class", "purity"}}. Otherwise mechanics are as below.

    With `link_policy="loss"`, before anything changes, each group's images in
    `val_loader` (joined by nearest group centre, or by label under `class`) are scored
    on the existing graph; a group links to an existing leaf per `choose_links`, and
    info gains `links` {group: {"leaf", "ratio"}} and `link_ratios` (every scored
    group's best candidate, linked or not). Only unlinked groups get fresh
    leaves (returned); each distinct linked leaf becomes one more router output, shared
    by the groups that link to it, and the router's pre-train labels and balance
    target follow those outputs.

    Cascade only: with `leaf_init="copy"` and classifier leaves, the new leaves are deep copies
    of the split leaf plus `noise` (no unit masking), unless the size check bumped the split
    (sizes then differ, so they stay fresh). Routers are always fresh.
    """
    old_step = model.steps[name] if hasattr(model, "steps") else None
    feats, hidden, reach, ys = _leaf_inputs(model, name, loader, device, limit_batches)
    grouped = class_groups(ys, reach, max_k) if policy == "class" else None
    extra = {"policy": policy}
    if grouped:
        groups, labels = grouped
        k, scores = len(groups), {}
        extra["groups"] = groups
    else:
        if policy == "class":
            extra = {"policy": "cluster", "policy_fallback": "no class >= 10% of reach"}
        k, labels, scores = pick_k(hidden, reach, max_k)
    sizes = [int((labels == c).sum()) for c in range(k)]
    info = {"cluster_sizes": sizes,
            "cluster_reach": [float(reach[labels == c].sum()) for c in range(k)],
            "router_pretrain_loss": None, "fallback": None,
            "k": k, "silhouettes": {str(c): v for c, v in scores.items()}, **extra}
    if silhouette_min is not None and hasattr(model, "steps"):
        pur = None
        if purity_gain is not None:  # labels (training batches) are read only for this check
            pur = (_purity_gain(ys, reach, labels, k, int(ys.max()) + 1), purity_gain)
        info["bump"] = _size_check(model, name, scores, sizes, silhouette_min, pur)
    if min(sizes) == 0:
        info["fallback"] = "copy"
        return split_leaf(model, name, noise), info
    links = {}
    if link_policy == "class":
        from beamroot.analysis import purity
        assign = (_class_assign(groups) if grouped
                  else _centre_assign(hidden, reach, labels, k))
        with torch.random.fork_rng(devices=[]):  # keep training's RNG stream as under `none`
            gmass, lmass = _val_class_mass(model, name, val_loader, device, limit_batches,
                                           assign, k)
        gclass = [purity(m)[0] for m in gmass]
        lstats = {n: purity(m) for n, m in lmass.items()}
        links = choose_class_links(gclass, lstats)
        info["link_candidates"] = {}
        for g, cls in enumerate(gclass):
            cands = [(p, n) for n, (c, p) in lstats.items() if cls is not None and c == cls]
            if cands:
                p, n = max(cands, key=lambda t: t[0])
                info["link_candidates"][str(g)] = {"leaf": n, "class": cls, "purity": p}
        info["links"] = {str(g): {"policy": "class", "leaf": leaf, "class": cls,
                                 "purity": pur}
                         for g, (leaf, cls, pur) in links.items()}
    if link_policy == "loss":
        assign = (_class_assign(groups) if grouped
                  else _centre_assign(hidden, reach, labels, k))
        with torch.random.fork_rng(devices=[]):  # keep training's RNG stream as under `none`
            group_losses, own = _val_group_losses(model, name, val_loader, device,
                                                  limit_batches, assign, k)
        links = choose_links(group_losses, own)
        best = {g: min(ls, key=ls.get) for g, ls in enumerate(group_losses) if ls and own[g] > 0}
        info["link_ratios"] = {str(g): {"leaf": leaf, "ratio": group_losses[g][leaf] / own[g]}
                               for g, leaf in best.items()}
        info["links"] = {str(g): {"leaf": leaf, "ratio": r} for g, (leaf, r) in links.items()}
    old = model.nodes[name]
    masks = neuron_masks(hidden, reach, labels, k)
    fresh_groups = [g for g in range(k) if g not in links]
    linked = list(dict.fromkeys(links[g][0] for g in sorted(links)))
    new = model.graph.split(name, len(fresh_groups))
    _set_steps(model, name, new)
    for leaf in linked:
        model.graph.add_edge(name, leaf)
    column = {g: i for i, g in enumerate(fresh_groups)}  # group -> router output
    column.update({g: len(fresh_groups) + linked.index(leaf) for g, (leaf, *_) in links.items()})
    model.nodes[name] = model.build_node(name).to(feats.device)
    for n, g in zip(new, fresh_groups):
        fresh = model.build_node(n).to(feats.device)
        if not hasattr(model, "steps"):
            fresh = _masked_leaf(old, fresh, masks[g], noise)
        elif (leaf_init == "copy" and model.leaf == "classifier"
              and model.steps[n] == old_step):
            fresh = copy.deepcopy(old)
            with torch.no_grad():
                for p in fresh.parameters():
                    p.add_(torch.randn_like(p) * noise)
        model.nodes[n] = fresh
    outputs = torch.tensor([column[g] for g in range(k)], device=labels.device)[labels]
    info["router_pretrain_loss"] = _pretrain_router(
        model.nodes[name], feats, outputs, reach, pretrain_steps, pretrain_lr)
    total = sum(info["cluster_reach"])
    if total > 0:
        shares = [0.0] * (len(fresh_groups) + len(linked))
        for g, r in enumerate(info["cluster_reach"]):
            shares[column[g]] += r
        model.graph.targets[name] = [r / total for r in shares]
    return new, info


def _centre_assign(hidden, reach, labels, k):
    """Group of an image = nearest reach-weighted group centre of the split leaf's
    hidden activations (cluster split policy)."""
    centres = []
    for c in range(k):
        wc = (labels == c).float() * reach
        centres.append((wc[:, None] * hidden).sum(0) / wc.sum().clamp(min=1e-12))
    centres = torch.stack(centres)
    return lambda h, y: torch.cdist(h, centres).argmin(1)


def _class_assign(groups):
    """Group of an image = the group holding its label (class split policy); -1 for
    classes in no group."""
    def assign(h, y):
        out = torch.full_like(y, -1)
        for i, g in enumerate(groups):
            for c in g:  # no torch.isin: not implemented on MPS
                out[y == c] = i
        return out
    return assign


def _val_group_losses(model, name, loader, device, limit_batches, assign, k):
    """Per split group, measured on the existing graph over the loader's batches (eval
    mode, no gradients): ({other leaf: mean NLL}, the split leaf's mean NLL). Images
    join a group by `assign(hidden activations of name, labels)`; means are weighted
    by each image's reach of `name`. A group with no reach gets ({}, 0.0)."""
    leaves = model.graph.leaves()
    was_training = model.training
    model.eval()
    groups, reach, nll = [], [], []
    with torch.no_grad():
        for x, y in islice(loader, limit_batches):
            x, y = x.to(device), y.to(device)
            out = model.route(x)
            groups.append(assign(model.nodes[name].hidden(model.features(x)), y))
            reach.append(out["log_reach"][name].exp())
            nll.append(torch.stack([-out["leaf_log_probs"][n].gather(1, y[:, None]).squeeze(1)
                                    for n in leaves], 1))
    model.train(was_training)
    groups, reach, nll = torch.cat(groups), torch.cat(reach), torch.cat(nll)
    group_losses, own = [], []
    for g in range(k):
        w = (groups == g).float() * reach
        if w.sum() <= 0:
            group_losses.append({})
            own.append(0.0)
            continue
        mean = ((w[:, None] * nll).sum(0) / w.sum()).tolist()
        losses = dict(zip(leaves, mean))
        own.append(losses.pop(name))
        group_losses.append(losses)
    return group_losses, own


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

    Removes the modules of `drop` and of every node left unreachable (and from a cascade's
    `frozen` set); returns their names.
    Routers keep their output width because redirected edges keep their column.
    """
    removed = model.graph.merge(keep, drop)
    for n in removed:
        del model.nodes[n]
        if hasattr(model, "frozen"):
            model.frozen.discard(n)
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


def node_param_groups(model, lr: float) -> list[dict]:
    """One param group per cascade node, in `model.nodes` order, each at `lr` and tagged
    with its node name (`group["node"]`) so its rate can be set on its own."""
    return [{"params": list(m.parameters()), "lr": lr, "node": n}
            for n, m in model.nodes.items()]


def set_router_lrs(model, opt: torch.optim.Optimizer, lr: float, decay: float,
                   epoch: int) -> None:
    """Router decay, before training `epoch`: an ancestor router's group (one with a
    `model.decay_start`) trains at `lr` * `decay` ** (epochs since it); every other group at `lr`."""
    for g in opt.param_groups:
        start = model.decay_start.get(g["node"])
        g["lr"] = lr if start is None else lr * decay ** (epoch - start)


def rebuild_optimizer(model, opt: torch.optim.Optimizer, lr: float | None = None
                      ) -> torch.optim.Optimizer:
    """New optimiser of the same type and hyperparameters over all current params.

    Params that survive the growth step keep their state (e.g. Adam moments);
    new params start with fresh state. The split leaf's old params are dropped.
    An optimiser with per-node groups (`node_param_groups`) gets per-node groups again,
    all at the base `lr`: a group's own lr may be decayed, so it is not the default.
    """
    defaults = {k: v for k, v in opt.param_groups[0].items() if k not in ("params", "node")}
    if "node" in opt.param_groups[0]:
        if lr is None:
            raise ValueError("per-node param groups need the base lr")
        defaults["lr"] = lr
        new = type(opt)(node_param_groups(model, lr), **defaults)
    else:
        new = type(opt)(model.parameters(), **defaults)
    for p in model.parameters():
        if p in opt.state:
            new.state[p] = opt.state[p]
    return new


def _ancestors(graph, name: str) -> set[str]:
    """Every node with a path to `name`, through all parents (the graph is a DAG)."""
    seen, stack = set(), [name]
    while stack:
        for p in graph.parents(stack.pop()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def save_graph_snapshot(model, run_dir, epoch: int) -> Path:
    path = Path(run_dir) / "graphs" / f"epoch_{epoch:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.graph.to_dict(), indent=2))
    return path


def grow(model, loader, device, epoch: int, run_dir, max_nodes: int,
         limit_batches=None, noise: float = 1e-3, max_k: int = 2,
         policy: str = "cluster", link_policy: str = "none",
         val_loader=None, silhouette_min=None, purity_gain=None,
         leaf_init: str = "fresh", route_acc_min: float = 0.93,
         branch_keep: float = 0.95) -> dict | None:
    """One growth step: cluster-split the leaf with the highest reach-weighted loss
    (ADR 0003), clustering the same batches the loss is measured on.

    The split is k-way for k up to `max_k` and up to what fits in `max_nodes` graph
    nodes; does nothing (returns None) if even a 2-way split would exceed it.
    Otherwise appends the event (with k, silhouettes, cluster sizes, router pre-train
    loss, any fallback and, under a link policy, `links`) to <run_dir>/growth.jsonl,
    writes a graph snapshot to <run_dir>/graphs/epoch_NNN.json, and returns the event.
    With `link_policy="loss"`, links are decided on `val_loader` (see `cluster_split`);
    `new_nodes` lists only the fresh leaves.

    A `TreeModel` grows by a supervised binary split instead (only `max_nodes`,
    `limit_batches`, `val_loader`, `route_acc_min` and `branch_keep` apply): the leaf with the highest
    reach-weighted loss among those with at least 2 classes is split on its confusion
    groups (from its validation confusion) at the earliest block whose probe head routes
    them with validation accuracy >= `route_acc_min`, else at the last block (fallback
    "last_block"). Probe heads train on frozen features of `limit_batches` training
    batches; the new router starts from the winning probe's weights. The event adds
    `groups`, `block`, `block_val_acc` (per probed block) and `fallback`. The root never
    falls back and never splits at its last block (DECISIONS Q5, `choose_root_block`); its
    events add `excluded_blocks: ["b4"]` (probed and logged, never chosen). If no other
    block reaches the bar, nothing splits and the returned
    and logged event is a miss, `{"epoch", "leaf", "skipped": "root_below_bar",
    "num_nodes", "leaf_losses", "groups", "block_val_acc", "excluded_blocks"}`, with no
    graph snapshot; the
    next growth step probes again. Each branch keeps, per conv, the fewest channels carrying
    `branch_keep` of its group's activation mass on the probes' training batches
    (`keep_channels`; 1.0 is a full copy); split events add `branch_keep`,
    `branch_channels` ({new leaf: kept channels per conv}) and `branch_macs` ({new leaf:
    its MACs}). The global RNG and the loaders' generators are left as
    they were.
    """
    if isinstance(model, TreeModel):
        return _grow_tree(model, loader, device, epoch, run_dir, max_nodes, limit_batches,
                          val_loader, route_acc_min, branch_keep)
    room = min(max_k, max_nodes - len(model.graph.nodes))
    if room < 2:
        return None
    losses = leaf_losses(model, loader, device, limit_batches)
    target = choose_leaf(losses)
    new, info = cluster_split(model, target, loader, device, noise, limit_batches,
                             max_k=room, policy=policy, link_policy=link_policy,
                             val_loader=val_loader, silhouette_min=silhouette_min,
                             purity_gain=purity_gain, leaf_init=leaf_init)
    event = {"epoch": epoch, "split": target, "new_nodes": list(new),
             "num_nodes": len(model.graph.nodes), "leaf_losses": losses, **info}
    if hasattr(model, "steps"):
        event["steps"] = {n: model.steps[n] for n in new}
        for n in (target, *new):  # the split leaf became a fresh router; the new leaves are new
            model.last_change_epoch[n] = epoch
        for n in _ancestors(model.graph, target):  # only the first split below sets it
            model.decay_start.setdefault(n, epoch)
    with open(Path(run_dir) / "growth.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")
    save_graph_snapshot(model, run_dir, epoch)
    return event
