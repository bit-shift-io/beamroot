# ADR 0003: Train on the path loss and split by clustering

**Status:** Accepted
**Date:** 2026-10-25

## Context
- Split routers sent each image about 50/50 to two near-copy leaves. Beam 1 could not explain an answer, and merges never fired.
- [Experiment 0001](../experiments/0001-routing-specialisation.md) compared two training losses (mixture, path) and three split initialisations (copy, random, cluster) on MNIST and CIFAR-10, 3 seeds each. It has the evidence for this decision.
- Beam-1 accuracy was a statistical tie across most variants, so it could not pick the winner.
- The project's priority is easily explainable graphs: decisive routers and pure leaves, a tree that groups classes level by level toward one leaf per class.

## Decision
- **DAG models train on the path loss** plus the load-balancing loss: the reach-weighted mean of each leaf's own cross-entropy. The soft prediction is still computed and reported as soft accuracy; it is no longer a training target.
- **Every split uses cluster initialisation:** reach-weighted 2-means on the split leaf's hidden activations; each new leaf keeps the units its group activates more (plus `--split-noise`) and re-initialises the rest; the new router is pre-trained on the group labels.
- **The copy split stays as a fallback only:** when 2-means leaves one group empty, the split copies the old leaf twice (plus noise). The event in `growth.jsonl` records `"fallback": "copy"`.
- The mixture training loss, `copy` and `random` as split modes, and the `--loss` / `--split-init` flags are deleted.

## Alternatives Considered
- **Mixture loss:** leaves are only judged as a blend, so they are not trained for the beam-1 setting. On CIFAR it left a soft − beam-1 gap of 0.016–0.024, against about 0.002 under the path loss.
- **Copy split:** under the path loss on MNIST its routers stayed undecided (entropy ≈ ln 2) with leaf purity about 0.21.
- **Random split:** decisive on MNIST, but the most seed-dependent variant on CIFAR (one router stayed undecided in one seed).
- **Keeping the modes behind flags:** more code paths to test and to keep compatible with growth and merges, for variants that lost on the criterion we care about.

## Consequences
- Cluster was the only initialisation with decisive routers under both losses and on both datasets, and gave the purest MNIST leaves (about 0.42 at 7 nodes, near the 4-leaf ceiling). A 19-node path + cluster run reached leaf purity 0.82 with six single-digit leaves at beam-1 0.9903.
- On CIFAR, path + cluster trails mixture + cluster on beam-1 (0.7365 vs 0.7444) and has the lowest soft accuracy of the grid. Accepted as the price of explainable routing.
- Each split costs about 3–4 s for clustering and router pre-training, instead of about 0.01 s for a copy.
- Old run directories still load: their `config.json` keeps `loss` / `split_init` keys, which nothing reads now. Experiment 0001's grid cannot be re-run with the current code.
- Next structural idea: k-way splits chosen by cluster count (see experiment 0001, "Not tried").
