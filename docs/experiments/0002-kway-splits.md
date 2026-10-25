# Experiment 0002: K-way splits and split policies

**Date:** 2026-10-25
**Status:** decided

## Hypothesis
Every split made two leaves, so natural 3-way groups were built as binary chains (experiment 0001, follow-up run). That wastes the node budget on extra routers. We expected:

- **`kway-cluster`** picks k = 2…4 by silhouette on the leaf's hidden activations. Where the activations hold more than two groups, it should build shallower graphs with more leaves for the same node budget, at no loss of beam-1 accuracy.
- **`kway-class`** gives each class with ≥ 10% of the leaf's mass its own child. It uses labels, so its leaves should be the purest. Its groups are not what the network finds natural, so it may cost accuracy.
- **`binary`** is today's behaviour and the baseline.

The main criterion is beam-1 test accuracy. Beam 3, leaf purity and split-router entropy break ties. Soft accuracy is reported, not ranked.

## Variants
- binary: `--max-k 2` (the current default)
- kway-cluster: `--max-k 4 --split-policy cluster`
- kway-class: `--max-k 4 --split-policy class`

## Setup
- **Command:** `.venv/bin/python -m beamroot.experiment splits`. Each run is `train.run` with `--model dag --merge --no-report`, plus the variant's flags.
- **Datasets:** MNIST (12 epochs) and CIFAR-10 (15 epochs), set per dataset by the grid.
- **Seeds:** 0, 1, 2. The seed also chooses the validation images.
- **Fixed flags:** `--max-nodes 19`, `--val-size 5000`, `--merge-threshold 0.05`, `--grow-every 1`, `--growth-batches 50`, `--split-noise 0.001`, `--balance-weight 0.1`, `--lr 0.001`, `--batch-size 128`. Silhouette margin `K_MARGIN = 0.05`; class threshold `CLASS_MIN_SHARE = 0.1`.
- **Data use:** splits read training batches; merges read the validation set. Accuracy, entropy and purity are measured on the test set and used for reporting only.
- **Device:** MPS. A smoke run came first: each variant once on MNIST, seed 0, `--epochs 3 --limit-batches 20` (`runs/experiments/splits-smoke/`). All three completed. In that short run `kway-cluster` never chose k > 2, so it matched `binary` exactly.
- **Code:** commit `dd5b879` plus uncommitted work for the flow-kway-linking feature: class flow, experiment variants, k-way cluster split, per-router balance targets, class split policy.
- **Balance targets:** every variant trains with per-router balance targets (each split router's target is its groups' reach shares). `binary` is therefore not byte-identical to experiment 0001's path + cluster.
- **Results:** `runs/experiments/splits/`. It holds 18 run folders, `summary.json` and `summary.md`. Log: `runs/experiments/splits.log`.
- **Wall clock:** 2026-10-25 11:05 to 17:46, about 6 h 41 min, exit 0. All 18 runs finished with `eval.json`. None failed.

## Results
Mean ± std over 3 seeds. Sorted by beam 1 within each dataset.

| dataset | variant | beam-1 acc | beam-3 acc | soft acc | soft − beam-1 | leaves | mean k | beam-1 nodes | split-router entropy | leaf purity |
|---|---|---|---|---|---|---|---|---|---|---|
| mnist | binary | 0.9911 ± 0.0007 | 0.9913 ± 0.0007 | 0.9913 ± 0.0007 | 0.0002 ± 0.0002 | 10.0 ± 0.0 | 2.00 | 4.53 ± 0.23 | 0.0182 ± 0.0015 | 0.8477 ± 0.0553 |
| mnist | kway-cluster | 0.9900 ± 0.0009 | 0.9903 ± 0.0011 | 0.9903 ± 0.0011 | 0.0003 ± 0.0004 | 12.0 ± 0.0 | 3.00 | 3.48 ± 0.10 | 0.0329 ± 0.0059 | 0.8924 ± 0.0237 |
| mnist | kway-class | 0.9896 ± 0.0027 | 0.9898 ± 0.0028 | 0.9898 ± 0.0028 | 0.0002 ± 0.0003 | 13.3 ± 0.6 | 3.47 | 3.47 ± 0.04 | 0.0343 ± 0.0119 | 0.9028 ± 0.0174 |
| cifar10 | binary | 0.7333 ± 0.0029 | 0.7376 ± 0.0052 | 0.7377 ± 0.0054 | 0.0044 ± 0.0026 | 10.0 ± 0.0 | 2.00 | 4.22 ± 0.08 | 0.0581 ± 0.0046 | 0.4290 ± 0.0267 |
| cifar10 | kway-cluster | 0.7333 ± 0.0029 | 0.7376 ± 0.0052 | 0.7377 ± 0.0054 | 0.0044 ± 0.0026 | 10.0 ± 0.0 | 2.00 | 4.22 ± 0.08 | 0.0581 ± 0.0046 | 0.4290 ± 0.0267 |
| cifar10 | kway-class | 0.7249 ± 0.0058 | 0.7304 ± 0.0074 | 0.7306 ± 0.0072 | 0.0057 ± 0.0017 | 14.0 ± 0.0 | 4.00 | 3.50 ± 0.10 | 0.0806 ± 0.0055 | 0.5620 ± 0.0251 |

Zero merges and zero links fired in all 18 runs. Training took 17.6–18.8 min per MNIST run and 21.4–21.7 min per CIFAR run; k-way splits added under a minute.

### Reading the beam-1 column
Two variants count as separated only when their mean gap exceeds the larger of their two stds (as in experiment 0001).

- **MNIST:** binary − kway-cluster = 0.0011 against a std of 0.0009: separated, but only just. binary − kway-class = 0.0015 against 0.0027: a tie. kway-cluster vs kway-class: a tie.
- **CIFAR-10:** binary and kway-cluster are the same runs in effect (identical numbers; see below). binary − kway-class = 0.0084 against 0.0058: separated.

So on beam 1 alone, `binary` ranks first on both datasets.

### How often k > 2 fired
- **kway-cluster, MNIST:** 9 of 15 splits chose k > 2. Every seed's first split (the root's children) chose k = 4. Per seed: [4, 2, 4, 3, 2], [4, 4, 2, 2, 3], [4, 4, 3, 2, 2].
- **kway-cluster, CIFAR-10:** 0 of 24 splits. The best silhouette was about 0.13–0.26 at every split, and no larger k beat k = 2 by the 0.05 margin. Each run made the same eight binary splits as `binary`, so the two variants produced identical results.
- **kway-class:** k = 4 at every split on CIFAR-10 (4 splits per run). On MNIST, 4-way at the first 3–4 splits and 2-way at the last two in seeds 1 and 2, when the node budget left room for only 2. The cluster fallback (no class ≥ 10%) never fired.

### Class flow, MNIST seed 0

**binary** — eight binary splits; the 4 / 1 / 9 / 7 group needs four levels, and leaf17 is a near-empty leftover:
```
root  reach 100%  {all classes}
├─ leaf0  reach 49%  {3,2,5,8,0,7,6}
│  ├─ leaf2  reach 21%  {2,3,7}
│  │  ├─ leaf4  reach 13%  2 (71%), 7 (27%)
│  │  └─ leaf5  reach 8%  3 (99%)  ← pure
│  └─ leaf3  reach 28%  {5,8,0,3,6}
│     ├─ leaf8  reach 17%  {8,0,6}
│     │  ├─ leaf10  reach 6%  0 (96%)  ← pure
│     │  └─ leaf11  reach 11%  8 (81%), 6 (11%)
│     └─ leaf9  reach 11%  5 (77%), 3 (20%)
└─ leaf1  reach 51%  {4,1,9,6,7,0}
   ├─ leaf6  reach 37%  {4,9,1,7}
   │  ├─ leaf12  reach 21%  {4,1}
   │  │  ├─ leaf14  reach 11%  {1}
   │  │  │  ├─ leaf16  reach 11%  1 (100%)  ← pure
   │  │  │  └─ leaf17  reach 0%  5 (42%), 2 (37%), 8 (10%), 7 (9%)
   │  │  └─ leaf15  reach 10%  4 (97%)  ← pure
   │  └─ leaf13  reach 16%  9 (60%), 7 (38%)
   └─ leaf7  reach 14%  6 (61%), 0 (30%)
```

**kway-cluster** — the two first-level routers split 4-way and 3-way at once; nine leaves are ≥ 93% one digit:
```
root  reach 100%  {all classes}
├─ leaf0  reach 52%  {3,2,8,5,7,0,9}
│  ├─ leaf2  reach 21%  {3,7,2,9}
│  │  ├─ leaf15  reach 10%  7 (74%), 9 (10%), 2 (10%)
│  │  └─ leaf16  reach 11%  3 (94%)
│  ├─ leaf3  reach 16%  {8,5}
│  │  ├─ leaf6  reach 7%  5 (97%)  ← pure
│  │  └─ leaf7  reach 9%  8 (93%)
│  ├─ leaf4  reach 9%  2 (98%)  ← pure
│  └─ leaf5  reach 6%  0 (98%)  ← pure
└─ leaf1  reach 48%  {4,1,6,9,0,7,5}
   ├─ leaf8  reach 21%  {4,9,7}
   │  ├─ leaf12  reach 10%  4 (99%)  ← pure
   │  ├─ leaf13  reach 3%  7 (99%)  ← pure
   │  └─ leaf14  reach 9%  9 (95%)
   ├─ leaf9  reach 9%  6 (99%)  ← pure
   ├─ leaf10  reach 11%  1 (99%)  ← pure
   └─ leaf11  reach 6%  0 (57%), 5 (28%), 2 (5%)
```

**kway-class** — three pure digits peel off at each router, and the rest go to a shared child that splits again:
```
root  reach 100%  {all classes}
├─ leaf0  reach 57%  {3,8,7,2,5,0,9}
│  ├─ leaf2  reach 9%  2 (98%)  ← pure
│  ├─ leaf3  reach 10%  3 (98%)  ← pure
│  ├─ leaf4  reach 9%  8 (100%)  ← pure
│  └─ leaf5  reach 28%  {7,5,0,9}
│     ├─ leaf14  reach 6%  0 (92%)
│     ├─ leaf15  reach 7%  5 (96%)  ← pure
│     ├─ leaf16  reach 10%  7 (97%)  ← pure
│     └─ leaf17  reach 5%  9 (73%), 1 (14%), 6 (10%)
└─ leaf1  reach 43%  {4,1,6,9,0,5,2}
   ├─ leaf6  reach 10%  1 (100%)  ← pure
   ├─ leaf7  reach 10%  4 (96%)  ← pure
   ├─ leaf8  reach 9%  6 (98%)  ← pure
   └─ leaf9  reach 14%  {9,0,5,2}
      ├─ leaf10  reach 4%  0 (100%)  ← pure
      ├─ leaf11  reach 2%  5 (98%)  ← pure
      ├─ leaf12  reach 6%  9 (100%)  ← pure
      └─ leaf13  reach 2%  2 (47%), 7 (24%), 8 (16%), 0 (7%)
```

## Per-variant notes

### binary (current default)
- **Good:** best beam-1 on both datasets, lowest split-router entropy (0.018 on MNIST), and the lowest seed spread on MNIST.
- **Bad:** deepest graphs: beam 1 evaluates 4.5 nodes on MNIST and 4.2 on CIFAR. Digit groups are built as chains (4 / 1 / 9 / 7 above takes four levels), and chains leave near-empty leftovers like leaf17 (reach 0%). Lowest MNIST purity (0.85).
- **Likely reason:** each split is the strongest 2-way cut, which suits accuracy, but natural 3- and 4-way groups cost an extra router per level.

### kway-cluster
- **Good:** on MNIST, the shallowest unsupervised graphs: 3.5 nodes per beam-1 path instead of 4.5, 12 leaves instead of 10 for the same 19-node budget, and purity 0.89 with tighter seed spread than binary (± 0.024 vs ± 0.055). Every seed's root split was 4-way.
- **Bad:** MNIST beam-1 0.0011 below binary, just past the one-std tie line (0.0009). On CIFAR it never fires, so it brings nothing there. Router entropy is higher than binary (0.033 vs 0.018) but still decisive.
- **Likely reason:** MNIST activations hold clear multi-digit groups once trained for 12 epochs; the 3-epoch smoke run did not. CIFAR-10 activations at these leaves never separate into more than two groups clearly enough to beat the 0.05 silhouette margin (silhouettes ≈ 0.13–0.26). The margin was not tuned.

### kway-class
- **Good:** purest leaves on both datasets (MNIST 0.90, CIFAR 0.56 vs 0.43) and the fewest beam-1 nodes (3.5 on both). On MNIST its flow tree is the most readable: three pure digits per router plus a shared rest child. Beam-1 ties binary on MNIST.
- **Bad:** CIFAR beam-1 0.0084 below binary, clearly separated, and the highest router entropy (0.081). Widest MNIST seed spread on beam 1 (± 0.0027).
- **Likely reason:** label groups are not the groups the network's features separate well. On CIFAR a router must make a 4-way cut along class lines its features may not support, which costs routing accuracy (higher entropy, lower beam 1). On MNIST the digits are separable enough that it does not matter. Which CIFAR classes the routers confuse was not checked.

## Not tried
- **Tuning `K_MARGIN`.** 0.05 was a guess. A smaller margin would make kway-cluster fire on CIFAR; whether that helps is unknown.
- **`--max-k` above 4.** kway-class hit the cap at nearly every split; MNIST might take 5-way root splits.
- **A larger node budget for k-way.** At 19 nodes kway-class sometimes had room only for 2-way final splits.
- **Validation-set split decisions.** Splits still read training batches; only links (experiment 0003) read the validation set.

## Decision
**Adopted: growth presets, `explainable` by default**, recorded in [ADR 0004](../adr/0004-growth-presets.md).

- `--preset explainable` (unsupervised, default) = `--max-k 4 --split-policy cluster`.
- `--preset pure` (supervised) = `--max-k 4 --split-policy class`.
- Explicit `--max-k` / `--split-policy` override the preset; `--max-k 2` gives binary splits. Every policy keeps its flag and tests.

**Why:** the one-std rule above ranked `binary` first, but that rule is too generous at 3 seeds. binary's MNIST lead over kway-cluster (0.0011, stds 0.0007 / 0.0009) gives t ≈ 1.7, p ≈ 0.17: not significant. On CIFAR-10 the two are identical. With accuracy tied, the explainability priority decides: kway-cluster makes MNIST beam-1 paths one node shorter and leaves purer. kway-class is kept as the `pure` preset for runs that may use labels, accepting its CIFAR-10 beam-1 cost (−0.0084, beyond noise). A binary "accurate" preset was dropped because the data does not support the accuracy claim.

**Follow-up:** tune `K_MARGIN` so k-way cluster can fire on CIFAR-10; experiment 0003 decides each preset's link policy.
