Status: pending
Complexity: 5 (design 2, scope 1, tests 1, hazards 1)
MCP: none

# Beam search inference and top-N paths

## What to build
- Beam search over the router graph: keep the N most probable partial paths per step; combine paths reaching the same node.
- `python -m beamroot.predict <run-dir> --beam 3 --samples 5` prints each image's true label and top-N paths (nodes, probability, leaf prediction).
- `python -m beamroot.evaluate <run-dir>` reports accuracy and mean nodes evaluated for beam 1, 3, 5 and soft routing.

## Files to create/modify
- beamroot/beam.py
- beamroot/predict.py
- beamroot/evaluate.py
- tests/test_beam.py

## Test approach
- Hand-built DAG with fixed routing probabilities: beam 1 picks the expected leaf; beam N returns paths in descending probability.
- Diamond DAG: shared node is evaluated once and its probability is the sum of both incoming paths.
- Large beam (≥ leaf count) matches soft-routing prediction.
- Nodes-evaluated count is correct on a known graph.

## Acceptance criteria
- [ ] Top-N output lists node path, path probability and predicted class.
- [ ] Evaluate prints a table: mode, accuracy, mean nodes evaluated.
- [ ] Beam ≥ leaf count equals soft-routing accuracy.

## Blocked by
02

## Gotchas
- With merges, "the path" to a node is ambiguous; report the highest-probability incoming path and the combined probability.
