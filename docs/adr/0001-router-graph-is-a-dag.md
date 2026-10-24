# ADR 0001: Router graph is a DAG

**Status:** Accepted
**Date:** 2026-10-24

## Context
- The model routes images through small router nodes to leaves.
- Early versions only grow by splitting, which produces a tree.
- Interpretability work shows features are shared across concepts, which a tree cannot express.

## Decision
- Nodes store a list of parents from the start. Beam search and soft routing treat the graph as a DAG.
- When paths reach the same node, their probabilities add and the node is evaluated once.

## Alternatives Considered
- Tree structure (single parent): simpler, but adding merges later would rewrite routing, beam search and serialisation.
- Fixed hand-designed DAG: removes the self-building aim of the project.

## Consequences
- Routing and beam search must handle shared nodes and topological ordering, even while the graph is a tree.
- Checkpoints serialise edges as a list, not nested children.
- Path explanations can become ambiguous once merges exist; reports show combined probability per node.
