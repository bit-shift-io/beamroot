# Experiment 0004: Cascade sizing

**Date:** 2026-10-30 to 2026-10-31
**Status:** pending user review

## Hypothesis
In the `dag` model every node reads one shared stem, so every image pays the full stem cost (about 3.8M of 4.76M MACs on an MNIST beam-1 path). The cascade model ([ADR 0005](../adr/0005-cascade-nodes-own-their-models.md)) gives each node its own small CNN at its own size step, so a coarse split near the root can be cheap and capacity grows only where a node cannot split its images. We expected a cascade variant to match `dag` accuracy at lower beam-1 MACs.

The main criterion is beam-1 test accuracy, with beam-1 MACs as the cost axis. Beam 3, leaf purity and bump counts break ties. Soft accuracy is reported, not ranked. There was one seed (0) on MNIST only; CIFAR-10 was not run (see Not tried).

## Variants
All cascade variants use `--start-step 1` (`START_STEP` in `beamroot/experiment.py`), `--max-nodes 19` unless stated, the `explainable` growth defaults, and default size-check thresholds (`--silhouette-min 0.1 --entropy-max 0.5 --spread-min 0.1`, purity check off).

Grid `cascade` (19 nodes):
- **dag:** `--model dag --merge --preset explainable`. The baseline, as in experiment 0003 `none`.
- **{resolution,hidden,both}-classifier:** `--model cascade --size-axis AXIS --leaf classifier`. Each leaf is a sized CNN head.
- **{resolution,hidden,both}-label:** `--leaf label`. A leaf is a learned class distribution with no network.

Grid `cascade29`: `dag-29` and `{resolution,hidden,both}-label-29`, as above with `--max-nodes 29`.

Grid `cascadeinit` (19 nodes, classifier leaves, axes `resolution` and `both`), three variants per axis:
- **-copy:** `--leaf-init copy`. New classifier leaves start as copies of the parent leaf plus `--split-noise`, as `dag` does. Routers stay fresh and pre-trained. A split that is bumped (sizes differ) and the empty-group fallback still build fresh leaves.
- **-settle:** `--settle-epochs 3 --bump-cooldown 2`. No splits or bumps in the last 3 epochs, and a router younger than 2 epochs (since creation or its last bump) is skipped by the decisiveness and spread checks. The two knobs are bundled, so their effects are not separated.
- **-copy-settle:** both.

## Setup
- **Commands:** `.venv/bin/python -m beamroot.experiment {cascade,cascade29,cascadeinit} --datasets mnist --seeds 0 --name {cascade,cascade29,cascadeinit}-mnist-s0`. Each run is `train.run` with `--no-report` plus the variant's flags.
- **Dataset:** MNIST, 12 epochs, seed 0, device MPS. Fixed flags as experiment 0003: `--val-size 5000 --grow-every 1 --growth-batches 50 --split-noise 0.001 --balance-weight 0.1 --lr 0.001 --batch-size 128`, plus cascade defaults `--prune-reach 0.01`.
- **Code:** commit `9665737`. The `cascade` and `cascade29` grids ran before `--leaf-init`, `--settle-epochs` and `--bump-cooldown` existed, so their runs are the `fresh`, 0, 0 defaults. Whether a default-flag run is bit-for-bit identical to the pre-flag code was not diffed; the existing cascade tests pass unchanged.
- **Results:** `runs/experiments/cascade-mnist-s0/`, `runs/experiments/cascade29-mnist-s0/`, `runs/experiments/cascadeinit-mnist-s0/` (each with `summary.json`, `summary.md`). Wall clock 5 h 0 min, 3 h 12 min and 7 h 3 min.
- **Not comparable:** `runs/experiments/smoke10/` (2-epoch smoke, made before the two fixes below) is not evidence.
- **Data use:** growth decisions read training batches (split clustering, size checks) and the validation set; accuracy, purity and entropy are measured on the test set for reporting only.
- **Two fixes before the grids** (found by probing the smoke grid, which had resolution variants near 0.30 and label variants near chance):
  - *Step-1 stem bottleneck.* A step-1 node on the `resolution` or `both` axis resizes MNIST to 7×7, then the stem pooled it twice to 1×1, leaving 16 features. Single-node MNIST (2 epochs, 10k images): resolution step 1 scored 0.610, step 2 0.872, step 4 0.936; both step 1 scored 0.409. The stem now drops its last max-pool at resolution 8 or below (`SMALL_RES = 8`), giving 144 features: resolution step 1 rose to 0.693, both to 0.633. Step 1 is still the weakest rung (0.874 after 10 epochs, step 2 reached 0.952).
  - *Label-leaf saddle.* Label leaves started exactly uniform, so all leaves were identical, every leaf had the same NLL for every image and the routers got zero gradient. With a fixed root and 2 label leaves, accuracy stayed at 0.114 and root usage at exactly [0.5, 0.5] for 6 epochs, at lr 1e-3 and 1e-2. Label-leaf logits now start at N(0, 0.5²) (`LABEL_INIT_STD`). A growth probe with that init (hidden axis, start step 2) reached 15 nodes and 0.595 accuracy by epoch 6 with decisive routers.
- **MPS bugs found on the way** (CPU tests missed all three): `adaptive_avg_pool2d` needs the input size divisible by the output size (28 → 21 crashed; replaced by an exact pooling-matrix `_avg_resize`), no float64 on MPS (`_purity_gain`), and no `Tensor.index_copy`.

## Results
Beam-1 test accuracy, one seed (no std). MACs are beam-1 MACs per image.

### Grid `cascade`: 19 nodes
| variant | beam-1 acc | beam-3 acc | leaves | leaf purity | beam-1 MACs | bumps | mean step | train min |
|---|---|---|---|---|---|---|---|---|
| dag | 0.9910 | 0.9913 | 12 | 0.919 | 4.76M | 0 | – | 18.4 |
| hidden-classifier | 0.9893 | 0.9895 | 11 | 0.843 | 11.92M | 19 | 2.06 | 35.8 |
| resolution-classifier | 0.9412 | 0.9414 | 10 | 0.630 | 5.71M | 18 | 2.00 | 30.2 |
| both-classifier | 0.9313 | 0.9341 | 11 | 0.772 | 4.08M | 18 | 1.95 | 29.3 |
| both-label | 0.7669 | 0.7683 | 10 | 0.782 | 7.34M | 21 | 2.11 | 55.2 |
| hidden-label | 0.6927 | 0.6931 | 10 | 0.743 | 12.60M | 21 | 2.11 | 58.3 |
| resolution-label | 0.5926 | 0.5945 | 10 | 0.699 | 5.22M | 20 | 2.05 | 39.3 |

### Grid `cascade29`: 29 nodes (label variants)
| variant | beam-1 acc | beam-3 acc | leaves | leaf purity | beam-1 MACs | bumps | mean step | train min |
|---|---|---|---|---|---|---|---|---|
| dag-29 | 0.9891 | 0.9894 | 18 | 0.880 | 4.83M | 0 | – | 23.9 |
| both-label-29 | 0.8206 | 0.8218 | 13 | 0.878 | 7.56M | 24 | 1.96 | 49.7 |
| hidden-label-29 | 0.6969 | 0.6983 | 13 | 0.756 | 13.83M | 24 | 1.96 | 62.6 |
| resolution-label-29 | 0.6804 | 0.6861 | 13 | 0.744 | 5.47M | 23 | 1.92 | 38.0 |

### Grid `cascadeinit`: leaf init and settle (19 nodes, classifier leaves)
Fresh rows are repeated from the `cascade` grid for comparison.

| variant | beam-1 acc | beam-3 acc | leaves | leaf purity | beam-1 MACs | bumps | mean step | train min |
|---|---|---|---|---|---|---|---|---|
| resolution-classifier (fresh) | 0.9412 | 0.9414 | 10 | 0.630 | 5.71M | 18 | 2.00 | 30.2 |
| resolution-classifier-copy | 0.9683 | 0.9686 | 11 | 0.643 | 6.00M | 18 | 1.95 | 46.2 |
| resolution-classifier-copy-settle | 0.9651 | 0.9655 | 11 | 0.720 | 3.57M | 12 | 1.63 | 61.9 |
| resolution-classifier-settle | 0.9578 | 0.9583 | 11 | 0.702 | 2.70M | 12 | 1.63 | 46.5 |
| both-classifier (fresh) | 0.9313 | 0.9341 | 11 | 0.772 | 4.08M | 18 | 1.95 | 29.3 |
| both-classifier-copy-settle | 0.9578 | 0.9601 | 12 | 0.812 | 1.35M | 8 | 1.42 | 88.4 |
| both-classifier-copy | 0.9540 | 0.9571 | 12 | 0.680 | 2.32M | 12 | 1.63 | 71.3 |
| both-classifier-settle | 0.9483 | 0.9522 | 11 | 0.810 | 2.85M | 12 | 1.63 | 76.9 |

## Per-variant notes
- **dag:** best accuracy per MAC. 0.991 at 4.76M MACs, no bumps, 12 leaves, 18 minutes. Nothing in the cascade family beats it on accuracy.
- **hidden-classifier:** matches `dag` (0.989) but costs 11.92M MACs, 2.5× more. The `hidden` axis keeps full-resolution input to the stem at every node, so the stem dominates and there is nothing to save. Not worth a further variant.
- **resolution-classifier / both-classifier (fresh):** cheap but 5–6 points below `dag` (0.941, 0.931). Leaf purity is low (0.63, 0.77). The likely reasons are fresh random leaves that train for few epochs, and 18 bumps per run that rebuild routers fresh and raise the mean step to about 2, which eats the savings.
- **label leaves (all three axes):** 0.59–0.77 at 19 nodes, purity only 0.70–0.78, far below every classifier variant. A label leaf answers one class for every image it receives, so a leaf with mixed classes is always wrong on its minority; only growth can fix that, and 19 nodes is the exact minimum for 10 pure binary leaves. All three stop at 10 leaves. At 29 nodes they gain 0.4–9 points (`both` 0.767 → 0.821, `resolution` 0.593 → 0.680, `hidden` flat at 0.693 → 0.697) and still stop at 13 leaves, not 18 like `dag-29`, with purity 0.74–0.88. More nodes does not close the gap, and `dag` itself gains nothing from them (0.991 → 0.989). They also cost more per image (3–5 node evaluations per training image, against 2.8 for `hidden-classifier`). Likely reason for the stall at 13 leaves is unverified: growth picks splits that do not separate classes cleanly. They keep an explainability appeal (the path is the explanation), but not at this accuracy. The user proposed dropping them.
- **-copy:** copying the parent leaf is the clearest accuracy gain: `resolution` +2.7 points (0.941 → 0.968), `both` +2.3 (0.931 → 0.954). It supports the idea that fresh leaves were under-trained. MACs are unchanged to slightly changed.
- **-settle (settle 3 + cooldown 2):** cuts bumps from 18 to 12 and the mean step from about 2.0 to 1.63, which more than halves MACs on `resolution` (5.71M → 2.70M, with accuracy 0.941 → 0.958). So premature bumps were inflating cost. The bundled knobs cannot be separated.
- **-copy-settle:** `both` reaches 0.958 at 1.35M MACs with only 8 bumps and mean step 1.42: 3.5× fewer MACs than `dag` for 3.3 points less accuracy. This is the first cascade variant that clearly beats `dag` on cost. On `resolution`, copy-settle (0.965) is not better than copy alone (0.968).
- **Training time:** wall-clock minutes went up in `cascadeinit` (46–88 min against about 30 for the fresh runs), not down. The cause is unexplained; the grid ran alone. Training cost matters if cost includes it.
- **ms/image:** not trusted. At batch size 1 on a GPU it mostly measures per-call overhead (dag 5 ms, cascade 4–12 ms).

## Not tried
- **CIFAR-10.** Skipped by the user's decision after the MNIST results (about 5 h per 7-variant grid on MNIST alone). The slice-10 acceptance criterion for a CIFAR-10 grid is therefore not met.
- **More seeds.** One seed only; MNIST seeds agree closely, but gaps of about 1 point (e.g. among the settle variants) are inside noise. The headline comparison (`dag` vs `both-classifier-copy-settle`) should get seeds 1 and 2.
- **`copy` and `settle` on the `hidden` axis,** and on the `both` axis with a cooldown value other than 2 or a settle other than 3.
- **Separating settle from cooldown,** and measuring when the bumps fall in the existing runs (late bumps are rebuilt fresh with no pre-training).
- **Why training time rose** with copy and settle.
- **A self-stopping growth rule** (stop when the worst leaf's loss is under a threshold or a split does not improve validation accuracy). `--max-nodes` already caps growth, so this was left as a separate design.
- **A hybrid leaf** (label leaf plus a tiny model on the node's hidden features) and tuning `LABEL_INIT_STD`: moot if label leaves are dropped.
- **A bigger step-1 node.** Step 1 (8 channels at 7×7) is still the weakest rung (0.874 single-node at 10 epochs, against 0.952 at step 2).
- **Verification gaps:** default-flag cascade runs were not diffed bit-for-bit against the pre-flag code; `--purity-check` was only smoked on synthetic data; the single-blob silhouette test forces the bump with `silhouette_min=0.99`, so default-0.1 behaviour on one blob is unchecked; the `--prune-reach`, `--silhouette-min`, `--entropy-max`, `--spread-min`, `--purity-gain`, `--settle-epochs`, `--bump-cooldown` and `--leaf-init` guards on non-cascade models compare against the default, so passing the default explicitly to a `dag` run is not rejected.

## Decision
Pending user review. Proposed, not yet adopted:
- Drop label leaves from the grids and write them off as the loser. The code and `--leaf label` flag stay for now so those runs stay reproducible (per the experiment-log memory, losers are written up before their code is deleted); delete only if the user decides to.
- Keep `--leaf-init copy`, `--settle-epochs` and `--bump-cooldown` as flags. Adoption of any of them as a default, and any change to ADR 0005 beyond the amendment note already added there, waits for more seeds.
- Next: seeds 1 and 2 for `dag` and `both-classifier-copy-settle`, then decide the defaults.
