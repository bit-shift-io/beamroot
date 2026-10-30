# ADR 0005: Cascade nodes own their models

**Status:** Accepted
**Date:** 2026-10-30

## Context
- In the `dag` model every node reads one shared stem's features. Each image pays the full stem cost (about 3.8M of about 4.8M MACs on an MNIST beam-1 path), whatever path it takes.
- Every node has the same capacity, whether it makes a coarse split near the root or a fine one deep down.
- The aim is a graph that separates coarse aspects first and finer details deeper, growing capacity only where a node cannot split its images.

## Decision
- A second model kind, `--model cascade`, sits beside `dag`. `dag` is unchanged and is the baseline.
- Each cascade node owns a full node model: it downsamples the raw image to its size step's resolution, runs its own conv stem, then a hidden layer and an output. There is no shared stem.
- Nodes still read the image, never a parent's output, so a node's behaviour does not depend on the path that reached it (as in ADR 0001).
- Node size follows a four-step ladder along a size axis (`resolution`, `hidden` or `both`). Step 4 equals today's node size. The root starts at the smallest step; children inherit their parent's step.
- Size checks (silhouette, decisiveness, spread, optional purity) bump a node one step up by rebuilding it with fresh random weights.
- New cascade nodes start random (routers are then pre-trained on the split groups); no leaf copies its parent.
- Cascade trains with pruned soft routing; merges and links are not supported.

## Alternatives Considered
- **One smaller shared stem:** cheap, but capacity does not grow with depth.
- **Multi-resolution shared stem computed lazily:** keeps sharing and saves inference, but nodes cannot differ in stem capacity, which is the idea under test.
- **Replacing `dag`:** loses the baseline; experiment variants must differ only in the thing compared.
- **Growing nodes in place (keeping weights):** rejected to avoid inherited bias.

## Consequences
- Training cost grows with node count, since each node runs its own stem. Pruned soft routing limits this once routers are decisive.
- Beam-1 inference cost depends on the steps along the path and can fall well below `dag`'s.
- Checkpoints carry a model kind and a step per node; `load_run` dispatches on the kind.
- Code that reads `model.stem` directly must go through `model.features(x)`.
- The glossary's Stem entry applies to `dag` only.
