# Experiment 0001: Routing specialisation

**Date:** 2026-10-25
**Status:** decided

## Hypothesis
Split routers sent each image about 50/50 to two near-copy leaves, so beam 1 could not explain an answer, and merges never fired. We expected:

- **Path loss** makes every leaf answer alone, which is the beam-1 setting. Routers should become decisive and the soft − beam-1 gap should shrink.
- **`random` split initialisation** breaks the symmetry between the two new leaves at once, at the cost of an accuracy dip after each split.
- **`cluster` split initialisation** starts each split already decisive: leaves specialise on 2-means groups of the old leaf's activations, and the new router is pre-trained on those groups.
- **Cross-depth merges** measured on a validation set give merging real candidates, so some merges should fire.

The main criterion is beam-1 test accuracy. Split-router entropy and leaf purity break ties. Soft accuracy is reported, not ranked.

## Variants
Two losses × three split initialisations = six variants.

- mixture + copy: `--loss mixture --split-init copy` (the old default)
- mixture + random: `--loss mixture --split-init random`
- mixture + cluster: `--loss mixture --split-init cluster`
- path + copy: `--loss path --split-init copy`
- path + random: `--loss path --split-init random`
- path + cluster: `--loss path --split-init cluster`

## Setup
- **Command:** `.venv/bin/python -m beamroot.experiment routing --name routing`, with all defaults. Each run is `train.run` with `--model dag --merge --no-report`.
- **Datasets:** MNIST (5 epochs) and CIFAR-10 (15 epochs), at the per-dataset default epochs.
- **Seeds:** 0, 1, 2. The seed also chooses the validation images.
- **Fixed flags:** `--val-size 5000`, `--merge-threshold 0.05`, `--max-nodes 7`, `--grow-every 1`, `--growth-batches 50`, `--split-noise 0.001`, `--balance-weight 0.1`, `--lr 0.001`, `--batch-size 128`.
- **Data use:** merges read the validation set. Splits read training batches. Accuracy, routing entropy and leaf purity are measured on the test set and used for reporting only.
- **Device:** MPS. Smoke runs came first: one variant per dataset (path + cluster, seed 0, `--epochs 3 --limit-batches 20`). Both completed.
- **Code:** commit `b6b6218` plus uncommitted work for the routing-specialisation feature: validation split, cross-depth merge, path loss, random and cluster split initialisation, and the experiment harness.
- **Results:** `runs/experiments/routing/`. It holds 36 run folders, `summary.json` and `summary.md`.
- **Wall clock:** 2026-10-24 19:24 to 2026-10-25 about 00:47, about 5 h 23 min, exit 0. All 36 runs finished with `eval.json` and `growth.jsonl`. None failed.

**How the columns are computed** (`beamroot/experiment.py` `run_metrics`):
- **beam-1 acc, soft acc:** test accuracy from `eval.json`.
- **split-router entropy:** mean routing entropy over every router except the root, measured on the test set. ln 2 ≈ 0.693 means undecided.
- **leaf purity:** mean over leaves of the leaf's most common class share. It is class purity on CIFAR too, not animal/vehicle. A leaf that sees half the classes evenly scores about 0.2.
- **closest merge KL:** the smallest symmetric KL between any merge-candidate pair, over every merge check in the run (validation set).
- **train min:** training time only. The measuring step is excluded.

## Results
The table shows mean ± std over 3 seeds, sorted by beam-1 within each dataset (`runs/experiments/routing/summary.md`).

| dataset | loss | split init | n | beam-1 acc | soft acc | soft − beam-1 | split-router entropy | leaf purity | merges | closest merge KL | train min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnist | path | copy | 3 | 0.9902 ± 0.0018 | 0.9914 ± 0.0014 | 0.0011 ± 0.0006 | 0.6928 ± 0.0001 | 0.2110 ± 0.0038 | 0.0 ± 0.0 | 1.9418 ± 0.4814 | 3.9 ± 0.0 |
| mnist | mixture | cluster | 3 | 0.9899 ± 0.0014 | 0.9904 ± 0.0014 | 0.0004 ± 0.0001 | 0.0478 ± 0.0066 | 0.4696 ± 0.0608 | 0.0 ± 0.0 | 11.9679 ± 1.7502 | 3.9 ± 0.0 |
| mnist | path | cluster | 3 | 0.9898 ± 0.0009 | 0.9902 ± 0.0009 | 0.0004 ± 0.0001 | 0.0380 ± 0.0210 | 0.4210 ± 0.0684 | 0.0 ± 0.0 | 2.9999 ± 0.4116 | 4.0 ± 0.0 |
| mnist | mixture | copy | 3 | 0.9894 ± 0.0008 | 0.9908 ± 0.0004 | 0.0014 ± 0.0003 | 0.6661 ± 0.0460 | 0.2213 ± 0.0147 | 0.0 ± 0.0 | 9.3625 ± 8.0264 | 3.8 ± 0.0 |
| mnist | path | random | 3 | 0.9892 ± 0.0020 | 0.9895 ± 0.0019 | 0.0004 ± 0.0002 | 0.0262 ± 0.0083 | 0.3994 ± 0.0206 | 0.0 ± 0.0 | 4.5025 ± 0.8049 | 3.9 ± 0.0 |
| mnist | mixture | random | 3 | 0.9866 ± 0.0010 | 0.9871 ± 0.0009 | 0.0006 ± 0.0002 | 0.1159 ± 0.0162 | 0.4486 ± 0.0774 | 0.0 ± 0.0 | 16.6779 ± 4.6960 | 3.8 ± 0.0 |
| cifar10 | mixture | cluster | 3 | 0.7444 ± 0.0057 | 0.7600 ± 0.0043 | 0.0155 ± 0.0016 | 0.1432 ± 0.0354 | 0.2760 ± 0.0425 | 0.0 ± 0.0 | 1.7698 ± 0.7446 | 12.7 ± 0.0 |
| cifar10 | path | copy | 3 | 0.7426 ± 0.0126 | 0.7453 ± 0.0130 | 0.0027 ± 0.0007 | 0.0885 ± 0.0228 | 0.2239 ± 0.0513 | 0.0 ± 0.0 | 0.4789 ± 0.2990 | 12.8 ± 0.0 |
| cifar10 | mixture | random | 3 | 0.7406 ± 0.0009 | 0.7630 ± 0.0076 | 0.0224 ± 0.0084 | 0.3139 ± 0.0793 | 0.2440 ± 0.0357 | 0.0 ± 0.0 | 3.4867 ± 2.1028 | 12.5 ± 0.0 |
| cifar10 | path | random | 3 | 0.7393 ± 0.0140 | 0.7423 ± 0.0158 | 0.0029 ± 0.0023 | 0.1622 ± 0.1920 | 0.2453 ± 0.0165 | 0.0 ± 0.0 | 0.9153 ± 0.6223 | 12.8 ± 0.0 |
| cifar10 | path | cluster | 3 | 0.7365 ± 0.0050 | 0.7387 ± 0.0053 | 0.0022 ± 0.0005 | 0.0196 ± 0.0001 | 0.2622 ± 0.0334 | 0.0 ± 0.0 | 0.6190 ± 0.4361 | 12.9 ± 0.0 |
| cifar10 | mixture | copy | 3 | 0.7362 ± 0.0070 | 0.7598 ± 0.0017 | 0.0236 ± 0.0073 | 0.6320 ± 0.0591 | 0.1637 ± 0.0259 | 0.0 ± 0.0 | 1.7960 ± 0.8213 | 12.6 ± 0.0 |

### Reading the beam-1 column
**Gaps under one std are ties.** Two variants count as separated only when their mean gap exceeds the larger of their two stds.

- **MNIST:** the top five variants span 0.9892–0.9902, and every gap among them is under one std (0.0008–0.0020). They are tied. Mixture + random (0.9866 ± 0.0010) is the only variant clearly behind: 0.0026 or more below each of the others. All variants clear the 97% beam-1 floor.
- **CIFAR-10:** mixture + cluster (0.7444 ± 0.0057) is clearly ahead of path + cluster (gap 0.0079) and mixture + copy (gap 0.0082). It ties with path + copy, mixture + random and path + random. All other pairs are ties.
- **Seed effect on CIFAR:** averaged over all six variants, beam-1 is 0.7344 for seed 0, 0.7398 for seed 1 and 0.7457 for seed 2. That spread (0.011) is larger than the spread between variant means (0.008). Path + copy and path + random get their high std from one strong seed-2 run each (0.7562 and 0.7536).

### Merge outcomes
**Zero merges fired in all 36 runs.** The closest pair ever measured was 0.136 symmetric KL (CIFAR path + copy, seed 2) against a threshold of 0.05. MNIST's closest was 0.355 (mixture + copy, seed 0). The data in each run's `growth.jsonl` explains why:

1. **Growth stops after epoch 2.** With `--max-nodes 7`, the splits at epochs 1 and 2 fill the budget: three routers and four leaves. After that only merges could change the structure.
2. **Early checks have no candidates.** At epoch 1 the only leaves are the root's two children, which are siblings. At epoch 2 the fresh split's leaves are protected (slice 02). At epoch 3 the epoch-2 leaves are still protected.
   - 33 of 36 runs grew a balanced tree (both root children split), so the epoch-3 check had no candidates.
   - 3 runs grew a chain, where the second split hit a grandchild: MNIST mixture + copy, + random and + cluster, all seed 0. Their epoch-3 check had one candidate, and beam 1 used 2.7 nodes on average instead of 3.
   - Real checks start at epoch 4 and there is no check after the last epoch. So MNIST gets one real check per run, and CIFAR gets eleven (epochs 4–14).
3. **Only leaf pairs from different subtrees are ever candidates.** The two inner routers are siblings in balanced trees, or ancestor and descendant in chains, so both are excluded. That leaves 4 cross-subtree leaf pairs (5 in chains). These leaves were trained on different images.
4. **The KL compares two leaves on every validation image, including images neither leaf receives.** Leaves that specialised on different images therefore disagree. Specialisation, the thing this experiment rewards, pushes the KL up.
5. **On CIFAR the gap widens over training.** In most runs the closest KL rises from epoch 4 to epoch 14. For example, path + copy seed 2 goes from 0.136 to 0.268, and path + cluster seed 2 from 0.151 to 0.592. More epochs make a merge less likely, not more.

Candidate KLs are much larger under the mixture loss than the path loss on both datasets: MNIST 9–17 vs 1.9–4.5, CIFAR 1.8–3.5 vs 0.5–0.9. Our guess is that a leaf trained only through the blend is less constrained on images routed elsewhere. This was not tested.

### Routing entropy: why path + copy is undecided on MNIST but not CIFAR
Every MNIST path + copy run ends with both split routers at 0.691–0.693, which is ln 2. On CIFAR the same variant ends at 0.03–0.12. Per-epoch `metrics.jsonl` (seed 0) shows how this happens:

- **MNIST:** split-router entropy sits at 0.688–0.693 from the split (epoch 2) to the end. Training cross-entropy is 0.065 at epoch 2 and 0.025 at epoch 5.
- **CIFAR:** both split routers start at 0.63–0.66. They fall to 0.31–0.43 at epoch 4, 0.17 at epoch 5 and about 0.05 by epoch 15. Training cross-entropy is still 1.1–0.8 over that period.

**Likely reason:** under the path loss, a router's gradient for an image is proportional to how much its two leaves' losses differ on that image. Copy-initialised leaves differ only by the 1e-3 split noise and their independent dropout masks.
- On MNIST both leaves are near zero loss on almost every image, so the difference, and the gradient, stays tiny. The router never leaves 50/50, and the two leaves stay copies.
- On CIFAR the losses are large enough for small leaf differences to matter. Once a router leans, the favoured leaf trains more on those images and the lean grows.

The mixture loss is undecided after copy splits on both datasets (MNIST 0.67, CIFAR 0.63). This is the symmetric trap: the mixture loss gives a router a gradient only where its children disagree, and near-copy leaves see the same images and stay copies.

The split-router entropy column leaves out the root. On CIFAR under the mixture loss, the root also stays soft: 0.30–0.65 on the test set, against 0.01–0.14 under the path loss in all but one run. The exception is path + copy seed 2: its root ended undecided (0.691), yet it was the best single CIFAR run (beam-1 0.7562).

### Soft − beam-1 gap: why mixture is much larger on CIFAR
On CIFAR the gap is 0.016–0.024 under the mixture loss and 0.002–0.003 under the path loss. On MNIST every gap is 0.0014 or less.

**Likely reason:**
- The mixture loss trains the soft prediction itself. Leaves only need to be right as a blend, so they can be individually weaker and complementary, like an ensemble. Beam 1 keeps only one leaf and loses that blend.
- The path loss trains each leaf to be right alone on the images it receives, which is exactly what beam 1 uses.
- On MNIST every leaf is already about 99% accurate alone, so blending has almost nothing to add.

The mixture loss also keeps CIFAR routers softer, including the root (see above), so beam 1 discards more of the mixture.

### Leaf purity
Leaf purity is 0.16–0.22 for every copy variant. That is about what two identical leaves sharing half the classes would give.

`random` and `cluster` roughly double purity on MNIST, to 0.40–0.47. Several MNIST runs grew a single-digit leaf with purity 0.95–0.99:
- mixture + random seed 0
- mixture + cluster seeds 0 and 2
- path + cluster seed 1

CIFAR purity stays low everywhere (0.16–0.28): leaves did not specialise on single classes in 15 epochs.

## Per-variant notes

### mixture + copy (old default)
- **Good:** high soft accuracy (MNIST 0.9908, CIFAR 0.7598). Beam 1 ties the MNIST leaders.
- **Bad:** split routers stay undecided (MNIST 0.666, CIFAR 0.632) and leaf purity is the lowest (CIFAR 0.164). Beam-1 is last on CIFAR (0.7362), clearly behind mixture + cluster. The CIFAR soft − beam-1 gap is the largest (0.024).
- **Likely reason:** the symmetric trap. Copy leaves see the same images and the mixture loss gives the router no reason to choose between them. On MNIST this costs nothing measurable because both copies are good alone.

### mixture + random
- **Good:** breaks the copy symmetry (MNIST entropy 0.116) and raises MNIST purity (0.449). It has the most stable CIFAR beam-1 (0.7406 ± 0.0009) and the highest CIFAR soft accuracy (0.7630).
- **Bad:** clearly last on MNIST beam-1 (0.9866), behind every other variant by more than one std. CIFAR routers are only half decided (0.314), and the CIFAR gap is large (0.022).
- **Likely reason:** re-initialised leaves restart from scratch after each split. MNIST has only 3 epochs after the last split to recover, and the mixture loss lets a weak leaf hide inside the blend rather than forcing it to be right alone. On CIFAR, 13 epochs is enough to recover. That fits the expected "accuracy dip after a split" trade-off.

### mixture + cluster
- **Good:** best CIFAR beam-1 mean (0.7444), clearly ahead of path + cluster and mixture + copy, and tied with the rest. It ties the MNIST leaders (0.9899) and has the highest MNIST purity (0.470) with decisive MNIST routers (0.048).
- **Bad:** the CIFAR soft − beam-1 gap is still 0.016, and CIFAR split routers are less decisive (0.143) than under any path variant except path + random. The CIFAR root stays soft (0.37–0.57).
- **Likely reason:** the pre-trained router and specialised leaves start decisive. The mixture loss does not undo that, but it does not push routing harder either, and it keeps training the blend rather than each leaf alone.

### path + copy
- **Good:** highest MNIST beam-1 mean (0.9902), though tied with four others. On CIFAR, routers become decisive without any initialisation help (0.089), the gap is small (0.003), and beam-1 is tied for first (0.7426).
- **Bad:** MNIST routing is fully undecided (0.6928 ± 0.0001, ≈ ln 2) with purity 0.211, so beam 1 there is a coin flip between two copies and explains nothing. Its CIFAR result has a large std (± 0.0126) because one seed was very strong.
- **Likely reason:** see "Routing entropy" above. The path loss breaks the copy symmetry only when leaf losses are large enough to differ (CIFAR), not on an easy task (MNIST). Its MNIST accuracy is high because the two copies are each a full, good leaf.

### path + random
- **Good:** decisive MNIST routers (0.026), MNIST purity 0.399, and a small gap on both datasets. CIFAR beam-1 is tied for first by mean (0.7393).
- **Bad:** the largest CIFAR std in both beam-1 (± 0.0140) and entropy (0.162 ± 0.192). In seed 2 one split router stayed undecided (0.679), and that run was also the variant's best (0.7536). MNIST beam-1 is tied but lowest of the tied group (0.9892).
- **Likely reason:** random leaves start different, so the path loss routes decisively from the start. Results depend heavily on the seed because the starting point is arbitrary.

### path + cluster
- **Good:** the most decisive routers in the grid (CIFAR 0.0196 ± 0.0001, MNIST 0.038), small gaps, MNIST purity 0.421, and a beam-1 tied with the MNIST leaders (0.9898). It has the lowest seed variance among CIFAR path variants.
- **Bad:** clearly behind mixture + cluster on CIFAR beam-1 (0.7365 vs 0.7444). It also has the lowest CIFAR soft accuracy (0.7387). Cluster splits cost about 3–4 s each, which is negligible here (train min 12.9 vs 12.5–12.8).
- **Likely reason:** cluster init fixes the partition early and the path loss hardens it. Each leaf then trains on fewer images. That trade may cost accuracy on CIFAR, where leaves are far from saturated. The 2-means groups are activation clusters, not classes (CIFAR purity 0.26), so the hard partition need not help classification.

## Follow-up run: path + cluster at 19 nodes
One exploratory MNIST run (seed 0, `--model dag --loss path --split-init cluster --merge --max-nodes 19 --epochs 12`, 2026-10-25). The goal was to see whether more leaves push the graph toward one leaf per digit.

- **Accuracy:** beam-1 0.9903 = soft (gap 0). That's no different from the 7-node grid (0.9898), so the gain is in explainability only.
- **Purity:** leaf purity 0.82, against 0.42 at 7 nodes. Every router is decisive (entropy ≤ 0.03), and no merge fired (closest KL 2.95).
- **Single-digit leaves:** 0, 3, 4, 6, 7 and 9 each got a leaf at 96–99% purity.
- **Mixed leaves:** four leaves are still mixed: {2, 7}, {5, 0}, {8, 1, 9}, and a leaf that is mostly 1. The digit 1 is scattered across four leaves.
- **Groupings follow appearance, not digit order:** root → {3, 8, 2, 5, 1, 7, 0} / {4, 6, 9, 0, 7, 1}; then {4, 9, 7}, {6, 0, 1}, {3, 2, 7}.
- **Evidence for k-way splits:** twice, a split's child was split again at once, building a natural 3-way group as a binary chain: {4, 9, 7} → 4 + {9, 7} → 9 + 7, and {6, 0, 1} → 6 + {0, 1} → 0 + 1. A 3-way split would have used fewer nodes and left budget for the mixed leaves.
- **Wall time:** 22 min including the report.

## Not tried
- **Per-image routing-entropy penalty (option C).** Deferred: the experiment first tested whether the loss and initialisation alone fix routing. The MNIST path + copy result (undecided routers under a beam-1-shaped loss) is the clearest case where it could still help.
- **Lowering the balance-loss weight (option B).** Not tested. The balance loss penalises batch-average usage, not per-image confidence, so it was judged not to be the cause (DECISIONS Q1).
- **Larger split noise (option A).** Covered by `random` init.
- **Grow-by-linking:** pointing a new split at an existing leaf instead of creating two new leaves. Deferred until results showed whether merges fire. They did not fire in any run, so this is now the obvious next structural idea.
- **Merge on class mix (option B in DECISIONS Q4):** comparing two nodes by the classes of the images they receive, instead of KL on every validation image. Deferred with grow-by-linking. The merge outcomes above suggest the KL criterion cannot fire once leaves specialise, which makes this worth revisiting.
- **Merging a router with a leaf:** out of scope.
- **K-way splits chosen by cluster count (follow-up idea):** let a split make 2–4 leaves instead of always 2, picking the count from how many clearly separate groups cluster init finds in the leaf's images. Could give shallower trees and splits that follow natural groups (e.g. {0, 6}, {1, 7}, {3, 5, 8}). `RouterNode` already takes any `num_children`; only the split and cluster init assume two. Open questions: a reliable rule for choosing k, the balance loss fighting lopsided k-way splits, and normalising routing entropy by ln k. Planned as a separate feature.
- **Higher merge threshold or more epochs after growth:** not run. The closest pairs were 0.14 (CIFAR) and 0.36 (MNIST) against 0.05, and CIFAR KLs grew with training.

## Decision
**Adopted: path loss + cluster split initialisation**, recorded in [ADR 0003](../adr/0003-path-loss-and-cluster-splits.md). The mixture training loss, `copy` and `random` as split modes, and the `--loss` / `--split-init` flags were deleted. `copy` survives only as cluster's fallback when 2-means finds an empty group. The soft prediction is still reported as soft accuracy.

Why the winner won:
- **Beam-1 accuracy could not decide.** Most variants were a statistical tie on beam-1 (see "Reading the beam-1 column").
- **Explainable graphs are the priority:** decisive routers and pure leaves, a tree that groups classes level by level toward one leaf per class. Cluster was the only initialisation with decisive routers under both losses and on both datasets, and it gave the purest MNIST leaves (about 0.42, near the 4-leaf ceiling).
- **The path loss trains for beam 1.** It trains each leaf to be right on the images it receives, which is exactly what beam 1 uses. Mixture left a soft − beam-1 gap of 0.016–0.024 on CIFAR, against about 0.002 under path.
- **It scales toward one leaf per class.** The 19-node follow-up run of path + cluster reached leaf purity 0.82, with six single-digit leaves at beam-1 0.9903 (see "Follow-up run").

Accepted cost: on CIFAR, path + cluster trails mixture + cluster on beam-1 (0.7365 vs 0.7444) and has the lowest soft accuracy in the grid.

Reproducing this grid needs the code from before the change: the `routing` grid and the `--loss` / `--split-init` flags no longer exist. The run directories under `runs/experiments/routing/` still load and evaluate.

**Follow-up:** k-way splits chosen by cluster count (see "Not tried"). The follow-up run built 3-way groups as binary chains twice, which is the direct evidence for it.
