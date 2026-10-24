# Glossary

- **Stem** — Shared conv layers every image passes through first. Not part of the router graph; never routed.
- **Router node** — Small MLP in the router graph that outputs a probability per child. Has one or more parents.
- **Leaf** — Graph node with no children; a small classifier head over the dataset's classes. Never routes.
- **Router graph** — The DAG of router nodes and leaves above the stem. One root. Always acyclic.
- **Path** — Sequence of nodes from root to a leaf. Its probability is the product of routing probabilities along it.
- **Beam** — Inference that keeps the N most probable partial paths at each depth. Paths reaching the same node combine.
- **Growth step** — A structural change to the router graph during training: a split or a merge.
- **Split** — Turning a leaf into a router node with two new leaves.
- **Merge** — Joining two behaviourally near-identical nodes into one that keeps both sets of parents.
- **Load-balancing loss** — Penalty pushing each router to use its children evenly across a batch.
- **Purity** — Share of the data reaching a node that belongs to its most common class or class group.
