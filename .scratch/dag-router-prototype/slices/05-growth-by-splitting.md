Status: pending
Complexity: 6 (design 2, scope 1, tests 1, hazards 2)
MCP: none

# Growth by splitting

## What to build
- Every `--grow-every K` epochs, split the leaf with the highest reach-weighted loss into a router with two new leaves, until `--max-nodes`.
- New leaves start as copies of the old leaf plus small noise, so accuracy does not drop sharply after a split.
- Growth events and graph snapshots are logged to the run directory.

## Files to create/modify
- beamroot/growth.py
- beamroot/models/graph.py (split op)
- beamroot/train.py (growth schedule, optimiser rebuild)
- tests/test_growth.py

## Test approach
- Split on a known graph: leaf becomes router, two new leaves, parents preserved, graph still acyclic.
- Highest-loss leaf is chosen given synthetic per-leaf losses.
- Soft output right after a split stays close to before (copied weights).
- Growth stops at `--max-nodes`.
- Checkpoint from a grown model reloads correctly.

## Acceptance criteria
- [ ] MNIST run grows to the node cap and reaches ≥97% at beam 1.
- [ ] `growth.jsonl` records epoch, split node and new node ids.
- [ ] Optimiser includes new node params after every split.

## Blocked by
02, 04

## Gotchas
- A split leaf may have multiple parents; the new router keeps all of them.
- Rebuilding the optimiser resets Adam state; keep existing params' state where possible or accept and document the reset.
