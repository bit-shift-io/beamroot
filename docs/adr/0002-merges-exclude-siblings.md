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
