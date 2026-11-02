# Experiment log

One file per experiment: `NNNN-short-name.md`, numbered in order. Write it when the runs finish, and fill in the Decision section once one is made. Never delete an entry, even when its idea lost: it records what not to retry.

ADRs record a decision. Entries here record the evidence and what was tried.

## Template

```
# Experiment NNNN: Title

**Date:** YYYY-MM-DD
**Status:** pending user review | decided

## Hypothesis
[What we expected, and why.]

## Variants
[Bullets: each variant and the flags that select it.]

## Setup
[Datasets, seeds, epochs, fixed flags, commit hash, results directory.]

## Results
[Table: mean ± std per variant and dataset. Main criterion first.]

## Per-variant notes
[For each variant: what it did well, what it did badly, the likely reason.]

## Not tried
[Ideas considered and left out, with the reason.]

## Decision
[What was adopted, why the winner won, and the ADR that records it.]
```

## Index

- [0001: Routing specialisation](0001-routing-specialisation.md): path vs mixture loss × copy/random/cluster split initialisation, with cross-depth merges. Decided: path loss + cluster split ([ADR 0003](../adr/0003-path-loss-and-cluster-splits.md)).
- [0002: K-way splits](0002-kway-splits.md): binary vs k-way cluster vs k-way class split policies at 19 nodes. Decided: `--preset {explainable,pure}`, default `explainable` = `--max-k 4 --split-policy cluster` ([ADR 0004](../adr/0004-growth-presets.md)).
- [0003: Link policies](0003-linking.md): link none vs loss vs class on the explainable preset. Decided: keep `none`; follow-up (looser `loss` ratio, `class` links under the `pure` preset) planned in the entry.
- [0004: Cascade sizing](0004-cascade-sizing.md): `dag` vs `--model cascade` (nodes with their own sized CNN) across size axes and leaf kinds, then leaf init and settle/cooldown. Pending user review: label leaves lost; `copy` init and `settle` make cascade 2–3.5× cheaper than `dag` at 3–4 points less accuracy; one seed, MNIST only.
- [0005: Router freeze](0005-router-freeze.md): seeds 0–2 of `dag` and `both-classifier-copy-settle`, plus `--freeze-routers quality` with and without `--router-lr-decay 0.5`. Freeze ties on beam-1 but loses 0.10 leaf purity; decay-freeze ties on both and saves ~16% of trained node evals. Neither adopted; 0004 headline holds over 3 seeds (−3.0 points, ~1.8× fewer MACs).
- [0006: Cascade on CIFAR-10](0006-cascade-cifar.md): Seed 0. Small-start cascades lose by 19–22 points at 2.1× the dag's MACs. A start-big (`-start4`) cascade ties the dag (0.735) at 4.2× the MACs, or about 0.9× once its routers are compacted. Compaction probes: big routers distil into step-1 students for 0.3–3 points, but a flat network of equal MACs (0.742) beats every routed model. Decision pending.
- [0007: Where can a CIFAR-10 network be split?](0007-where-to-split.md): Off-repo probes, seed 0. Vehicles against animals is routable from the first conv block (91%). Trees with a shared trunk, a supervised group router and per-group branches tie a plain network at equal MACs when split at b2 or later, and beat a no-routing network of equal reduced MACs by 2.2–2.7 points. Early exit saves 19–32% for 0–0.25 points. Decision pending.
- [0008: Trunk trees on CIFAR-10](0008-trunk-tree-cifar.md): Seed 0, `--model tree` grown in `deep4`. At equal epochs, `tree` ties a plain network within noise (−1.1 points) at 31% fewer MACs; `tree-keep1` ties at equal MACs; `tree-progressive` loses by 4.9. Every tree beats `dag`. The animal half (60% of images) never splits. Decision pending; pruned-plain control and LR schedule fix next.
