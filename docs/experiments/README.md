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
