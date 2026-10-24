Status: done
Complexity: 3 (design 1, scope 1, tests 1, hazards 0)
MCP: none

# Load-balancing loss and routing stats

## What to build
- Per-router load-balancing loss added to training loss, scaled by `--balance-weight` (0 disables).
- Per-epoch routing stats in `metrics.jsonl`: each router's mean child usage and routing entropy.

## Files to create/modify
- beamroot/losses.py
- beamroot/models/dag.py (expose per-router probabilities)
- beamroot/train.py
- tests/test_losses.py

## Test approach
- Loss is zero for uniform child usage and grows as usage skews.
- Loss is weighted by reach probability so unreachable routers contribute nothing.
- Stats appear in metrics after a smoke run.

## Acceptance criteria
- [ ] With `--balance-weight 0`, behaviour matches slice 02.
- [ ] With default weight, no router sends >90% of data to one child after training on MNIST.

## Blocked by
02
