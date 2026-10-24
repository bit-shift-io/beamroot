Status: pending
Complexity: 7 (design 2, scope 1, tests 2, hazards 2)
MCP: none

# Merging near-identical nodes

## What to build
- With `--merge`, each growth step also checks node pairs at the same depth for near-identical outputs on a held-out batch.
- Pairs below `--merge-threshold` (symmetric KL) merge: the survivor keeps both parent sets; the other node and its now-unreachable descendants are removed.
- Diagram and report mark merged nodes and show their multiple parents.

## Files to create/modify
- beamroot/growth.py (merge candidate search, merge op)
- beamroot/models/graph.py (merge op, prune unreachable)
- beamroot/report.py
- tests/test_merge.py

## Test approach
- Two leaves with identical weights merge; parents are combined; graph stays acyclic.
- Merge never creates a cycle, even when one candidate is an ancestor of the other (such pairs are skipped).
- Soft output after merging stays close to before.
- Pruning removes unreachable nodes and their params from the optimiser.

## Acceptance criteria
- [ ] MNIST run with `--merge` completes and the report shows ≥1 multi-parent node, or logs that no pair met the threshold.
- [ ] `growth.jsonl` records merge events.
- [ ] Without `--merge`, behaviour matches slice 05.

## Blocked by
05, 06

## Gotchas
- Two parents may both route to the survivor; combine the edges rather than duplicating them.
- Restrict candidates to nodes of the same kind (leaf with leaf, router with router).
