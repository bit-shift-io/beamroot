# ADR 0002: Merges exclude siblings

**Status:** Accepted
**Date:** 2026-10-24

## Context
- Growth splits a leaf into a router with two near-identical leaves (copies plus 1e-3 noise).
- Merging joins near-identical nodes at the same depth and of the same kind (ADR 0001).
- Fresh siblings are the closest pairs in the graph, so they would always be merged first.

## Decision
- Nodes that share a parent are never merge candidates, and neither are ancestor/descendant pairs.

## Alternatives Considered
- Merge siblings by folding the router's two output columns into one. This changes the router's soft output, since logsumexp cannot be expressed as one linear column. It also leaves a one-output pass-through router, which undoes the split, so the graph would loop between splitting and merging.

## Consequences
- With split-only growth the graph is a chain, so the only candidates are cousins. A default 7-node MNIST run has none. At 9 nodes the cousin pairs measured ~13 nats of symmetric KL against a 0.05 threshold.
- In practice `--merge` never produces a multi-parent node. Making merges happen needs a growth policy that creates cross-branch similarity (for example wider splits or re-routing), not a lower threshold.
- Merge candidates are scored on the test loader, because there is no validation split yet.

## Update 2026-10-24: same-depth rule removed
- Merge candidates no longer need the same depth. Pairs must still be the same kind, share no parent, and not be ancestor and descendant.
- Why loops cannot form: a merge redirects each edge `p -> drop` to `p -> keep`. That edge closes a loop only if `p` is `keep` or a descendant of `keep`. In both cases `drop` (a child of `p`) would be a descendant of `keep`, and ancestor/descendant pairs are excluded. Depth plays no part in this argument, so the graph stays acyclic (ADR 0001). `RouterGraph.merge` still checks all of these rules itself.
- Depth is longest-path depth. A cross-depth merge moves the kept node and its subtree deeper. Nothing caches depths, so later steps always see current depths.
- Nodes created by the latest growth step (the previous step's split, because merges run before the split) are never candidates. A fresh leaf is a near-copy of the leaf it came from, so letting it merge straight away would undo the split.
- Merge candidates are now scored on the validation set, not the test loader.
