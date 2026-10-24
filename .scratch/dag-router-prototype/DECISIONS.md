# Decisions

### Q1: Prototype goal
**Decision:** Prove the mechanics first; report self-organisation; compare against a baseline.
- **Why:** User wants all three; mechanics must work before the other two mean anything.
- **Implication:** Accuracy target is "reasonable" (≥97% MNIST), not state of the art.
- **Alternatives considered:** Lead with competitiveness or emergent grouping — deferred to later slices.

### Q2: Tree or DAG
**Decision:** DAG data structure from day one; growth only splits until the merge slice.
- **Why:** Shared features suit a DAG; split-only growth keeps early debugging simple.
- **Implication:** Beam search must combine paths that reach the same node.
- **Alternatives considered:** Split+merge from start (two rules to debug at once); fixed hand-built DAG (loses self-building).

### Q3: Interface
**Decision:** Python package + CLI scripts; artefacts saved per run (PNG graph diagram, HTML report).
- **Why:** Repeatable and testable; report covers the self-organisation goal.
- **Alternatives considered:** Jupyter notebook — can be added later on top of the package.

### Q4: Leaf output
**Decision:** Each leaf is a small classifier head over the dataset's classes.
- **Why:** Gives each path a concrete answer; final prediction is the path-probability-weighted mix of leaves.

### Q5: Training vs inference routing
**Decision:** Soft routing in training (all branches run, weighted); beam search at inference.
- **Why:** Hard choices block gradients; soft routing is the standard fix (soft decision trees).
- **Alternatives considered:** Gumbel-softmax / straight-through — more complex, revisit if soft/hard gap is large.

### Q6: Growth rule
**Decision:** Every K epochs, split the leaf with the highest summed loss contribution into a router with two new leaves, up to `--max-nodes`.
- **Why:** Simple, deterministic, targets where the model is worst.
- **Implication:** Optimiser is rebuilt after each growth step.
- **Alternatives considered:** Split on activation multimodality (clustering) — candidate for later.

### Q7: Collapse prevention
**Decision:** Load-balancing loss on each router: penalise deviation of mean child usage from uniform, weighted by `--balance-weight`.
- **Why:** Routers otherwise send all data down one branch.

### Q8: Merge rule
**Decision:** Merge two nodes when their outputs on a held-out batch are near-identical (symmetric KL below `--merge-threshold`); survivor inherits both parent sets.
- **Why:** Cheap behavioural similarity test; produces multi-parent nodes.

### Q9: Environment
**Decision:** Plain `venv` + `pip install -e .[dev]`, PyTorch, torchvision, pytest, networkx, matplotlib.
- **Why:** `uv` is not installed; Python 3.14 is present.

### Q10: Baseline
**Decision:** Plain CNN, width chosen so parameter count is within 10% of a given DAG run (`--match-params <run-dir>`).
- **Why:** Fair comparison against the grown model's final size.

## Assumptions
- PyTorch wheels exist for Python 3.14 on macOS; otherwise fall back to Python 3.12/3.13.
- MPS is available; CPU fallback is acceptable for tests.

## Trade-offs considered
- Soft routing costs full compute in training; savings only appear at inference.
- Split-by-loss may over-split noisy leaves; `--max-nodes` caps it.

## New CONTEXT.md terms
- Stem, Router node, Leaf, Router graph, Path, Beam, Growth step, Split, Merge, Load-balancing loss, Purity.
