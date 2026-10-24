Status: pending
Complexity: 6 (design 2, scope 1, tests 1, hazards 2)
MCP: none

# Router graph with soft routing

## What to build
- A DAG-based model: stem → router graph → leaves. Starts as root router with two leaves.
- Soft forward pass: each node's reach probability = sum over parents of parent reach × routing probability; output = reach-weighted mix of leaf predictions.
- `--model dag` trains it on MNIST via the same train CLI and saves a checkpoint that restores the graph.

## Files to create/modify
- beamroot/models/graph.py (node registry, edges as list, topological order, add/remove ops)
- beamroot/models/nodes.py (RouterNode MLP, Leaf head)
- beamroot/models/dag.py (stem + graph, soft forward)
- beamroot/train.py (model selection)
- tests/test_graph.py
- tests/test_dag_forward.py

## Test approach
- Graph rejects cycles; topological order is valid on a hand-built multi-parent DAG.
- Reach probabilities sum to 1 across leaves on a multi-parent DAG.
- Soft forward output is a valid distribution; gradients reach the stem and every router.
- Checkpoint round trip restores identical outputs.

## Acceptance criteria
- [ ] Nodes hold a parent list per ADR 0001.
- [ ] A hand-built diamond DAG (root → A, B → shared leaf) produces correct reach probabilities.
- [ ] `--model dag` trains on MNIST and reaches ≥95% with the fixed 3-node graph.
- [ ] Checkpoints store graph structure plus weights.

## Blocked by
01

## Gotchas
- Nodes added later must be registered in an `nn.ModuleDict` so their params reach the optimiser and `state_dict`.
- Use log-space or clamp when multiplying probabilities down deep paths.
- Leaf and router heads read stem features, not parent outputs; keep that fixed so merges stay simple.
