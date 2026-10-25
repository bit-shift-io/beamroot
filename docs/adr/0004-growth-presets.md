# ADR 0004: Growth presets, explainable by default

**Status:** Accepted
**Date:** 2026-10-25

## Context
- Splits can now make 2–4 children (`--max-k`) and group a leaf's images two ways (`--split-policy cluster` or `class`).
- [Experiment 0002](../experiments/0002-kway-splits.md) compared binary, k-way cluster and k-way class splits on MNIST and CIFAR-10 at 19 nodes, 3 seeds each.
- Binary led k-way cluster on MNIST beam 1 by 0.0011 (stds 0.0007 / 0.0009; t ≈ 1.7, p ≈ 0.17 with 3 seeds), which is not a significant margin. On CIFAR-10 k-way cluster never chose k > 2, so the two were identical.
- The project's priority is explainable graphs: decisive routers, pure leaves, short beam-1 paths. On MNIST k-way cluster cut beam-1 path length from 4.5 to 3.5 nodes and raised leaf purity from 0.85 to 0.89.
- K-way class gave the purest leaves (MNIST 0.90, CIFAR-10 0.56 vs 0.43) but lost 0.008 beam 1 on CIFAR-10, beyond seed noise. It uses labels, which some runs want and others do not.

## Decision
- **`--preset {explainable,pure}`** sets the growth defaults:
  - `explainable` (unsupervised, the default): `--max-k 4 --split-policy cluster`.
  - `pure` (supervised): `--max-k 4 --split-policy class`.
- An explicit `--max-k` or `--split-policy` overrides the preset. Binary splits stay available as `--max-k 2`; there is no binary preset.
- Every policy keeps its flag, code and tests (`docs/memory/growth-policies-stay.md`).

## Alternatives Considered
- **Keep binary as the default:** ranked first by the beam-1 rule, but by a margin within seed noise on MNIST and none on CIFAR-10. It builds class groups as binary chains, which lengthens beam-1 paths.
- **A three-way preset including `accurate` (binary):** dropped. The accuracy claim is not supported by the data, and `--max-k 2` already selects binary.
- **`pure` as the default:** most readable leaves, but uses labels and costs CIFAR-10 accuracy beyond noise.

## Consequences
- Default growth changes: a run without `--max-k` now clusters for k = 2…4 at each split. Splits cost a little more (extra k-means and silhouette passes; under a minute per run in experiment 0002).
- On CIFAR-10 the default behaves as binary until the silhouette margin (`K_MARGIN = 0.05`) is tuned or features separate better.
- `config.json` records `preset` and the resolved `max_k` / `split_policy`. Old run directories lack `preset`; nothing reads it back.
- Link policies (experiment 0003) will extend the presets: the link default for each preset is decided there.
