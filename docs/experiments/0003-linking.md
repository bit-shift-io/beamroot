# Experiment 0003: Link policies

**Date:** 2026-10-26
**Status:** decided

## Hypothesis
A split always made fresh leaves, even when an existing pure leaf already handled some of the split leaf's images (e.g. the 4s in a mixed leaf, when a pure 4 leaf exists). Merges never fired, so the graph stayed a tree. We expected:

- **`loss`** links a split group to an existing leaf whose validation NLL on the group is under half the split leaf's. Unsupervised. It should save nodes and send each class to one leaf.
- **`class`** links a group to an existing leaf with the same dominant class and validation purity ≥ 0.9. Supervised.
- Shared leaves combine path probability, so **beam 3 should gain over beam 1**, and graphs with shared leaves might let **KL merges** fire.

The main criterion is beam-1 test accuracy. Beam 3, leaf purity and split-router entropy break ties. Soft accuracy is reported, not ranked.

## Variants
All on the `explainable` preset chosen by experiment 0002 (`--max-k 4 --split-policy cluster`).

- none: `--link-policy none` (the current default)
- loss: `--link-policy loss` (`LINK_RATIO = 0.5`)
- class: `--link-policy class` (`LINK_PURITY = 0.9`)

## Setup
- **Command:** `.venv/bin/python -m beamroot.experiment links`. Each run is `train.run` with `--model dag --merge --no-report`, plus the variant's flags.
- **Datasets:** MNIST (12 epochs) and CIFAR-10 (15 epochs). **Seeds:** 0, 1, 2.
- **Fixed flags:** as experiment 0002: `--max-nodes 19`, `--val-size 5000`, `--merge-threshold 0.05`, `--grow-every 1`, `--growth-batches 50`, `--split-noise 0.001`, `--balance-weight 0.1`, `--lr 0.001`, `--batch-size 128`.
- **Data use:** link decisions read the validation set; split clustering reads training batches. Accuracy, entropy and purity are measured on the test set for reporting only.
- **RNG isolation:** link scoring iterates the validation loader, which used to advance the global RNG and change every later random draw. It now runs inside `torch.random.fork_rng`, so a link variant where no link fires trains bit-for-bit like `none`. The first smoke (`runs/experiments/links-smoke/`) predates this fix and is not comparable; `runs/experiments/links-smoke2/` is (3 epochs, 20 batches: all three variants identical, no links).
- **Code:** commit `dd5b879` plus uncommitted flow-kway-linking work (slices 01–09).
- **Results:** `runs/experiments/links/` (18 runs, `summary.json`, `summary.md`). Log: `runs/experiments/links.log`.
- **Wall clock:** 2026-10-25 19:03 to 2026-10-26 01:45, about 6 h 42 min, exit 0. All 18 runs finished with `eval.json`.

## Results
Mean ± std over 3 seeds. Sorted by beam 1 within each dataset.

| dataset | variant | beam-1 acc | beam-3 acc | soft acc | soft − beam-1 | leaves | mean k | links | beam-1 nodes | leaf purity |
|---|---|---|---|---|---|---|---|---|---|---|
| mnist | class | 0.9902 ± 0.0007 | 0.9906 ± 0.0004 | 0.9906 ± 0.0004 | 0.0004 ± 0.0003 | 11.7 ± 0.6 | 2.89 | 1.0 ± 1.0 | 3.52 ± 0.11 | 0.8816 ± 0.0139 |
| mnist | none | 0.9900 ± 0.0009 | 0.9903 ± 0.0011 | 0.9903 ± 0.0011 | 0.0003 ± 0.0004 | 12.0 ± 0.0 | 3.00 | 0.0 | 3.48 ± 0.10 | 0.8924 ± 0.0237 |
| mnist | loss | 0.9900 ± 0.0009 | 0.9903 ± 0.0011 | 0.9903 ± 0.0011 | 0.0003 ± 0.0004 | 12.0 ± 0.0 | 3.00 | 0.0 | 3.48 ± 0.10 | 0.8924 ± 0.0237 |
| cifar10 | none | 0.7333 ± 0.0029 | 0.7376 ± 0.0052 | 0.7377 ± 0.0054 | 0.0044 ± 0.0026 | 10.0 ± 0.0 | 2.00 | 0.0 | 4.22 ± 0.08 | 0.4290 ± 0.0267 |
| cifar10 | loss | identical to none | | | | | | 0.0 | | |
| cifar10 | class | identical to none | | | | | | 0.0 | | |

`none` reproduces experiment 0002's `kway-cluster` numbers exactly.

### How often links fired
- **loss: 0 links in 9 runs.** The lowest ratio seen per run (best existing leaf's loss ÷ split leaf's loss) was 0.94, 1.33, 1.30 on MNIST and 1.03, 0.98, 0.97 on CIFAR-10, against a threshold of 0.5. The split leaf was trained on exactly the images being scored, so another leaf almost never beats it, let alone by half.
- **class: 3 links in 9 runs, all on MNIST** (seed 0: two, seed 2: one; seed 1: none). On CIFAR-10 no group's dominant class matched a leaf with purity ≥ 0.9; the best same-class candidates were 0.23–0.43 pure.
- Each link saved a node, which the budget spent on one extra split (6 splits instead of 5 in those runs).

### Do links raise beam 3 over beam 1?
No measurable effect. MNIST beam-3 − beam-1 is 0.0004 under `class` vs 0.0003 under `none`. Only 1 link per run on average, so few paths share a leaf.

### Do merges fire now?
No. Zero merges in all 18 runs, as in experiments 0001 and 0002.

### Class flow with links, MNIST seed 0, `class`
Both links are the case the feature was built for: the 7s in leaf8's group go to the existing 7 leaf (leaf13), and the 0s in leaf11 go to the pure 0 leaf (leaf5), instead of fresh copies.
```
root  reach 100%  {all classes}
├─ leaf0  reach 55%  {3,8,2,5,7,0,9}
│  ├─ leaf2  reach 20%  {3,7,9}
│  │  ├─ leaf15  reach 10%  3 (97%)  ← pure
│  │  └─ leaf13  reach 12%  7 (84%), 9 (14%)
│  ├─ leaf3  reach 19%  {8,5}
│  │  ├─ leaf6  reach 8%  5 (96%)  ← pure
│  │  └─ leaf7  reach 11%  8 (85%), 6 (8%)
│  ├─ leaf4  reach 10%  2 (98%)  ← pure
│  └─ leaf5  reach 9%  0 (99%)  ← pure
└─ leaf1  reach 45%  {4,1,6,9,0,7,5}
   ├─ leaf8  reach 20%  {4,9,7}
   │  ├─ leaf12  reach 10%  4 (99%)  ← pure
   │  ├─ → leaf13  (shared, see above)
   │  └─ leaf14  reach 8%  9 (96%)  ← pure
   ├─ leaf9  reach 9%  6 (97%)  ← pure
   ├─ leaf10  reach 11%  1 (99%)  ← pure
   └─ leaf11  reach 6%  {0}
      ├─ leaf16  reach 3%  5 (32%), 2 (30%), 0 (22%), 8 (7%)
      └─ → leaf5  (shared, see above)
```
leaf13 was 98% pure on the validation set when linked; after training on with both parents it is 84% 7 and 14% 9 on test.

No `loss` run made a link, so there is no `loss` tree to quote.

## Per-variant notes

### none (current default)
- **Good:** simplest; the same results as experiment 0002's kway-cluster.
- **Bad:** the graph stays a tree, so a class split across two branches gets two leaves (MNIST seed 0 has two 0-leaves: leaf5 and part of leaf11).

### loss
- **Good:** unsupervised; costs nothing when it does not fire (bit-identical to `none`).
- **Bad:** never fired. At `LINK_RATIO = 0.5` it cannot fire in practice: ratios bottomed out at 0.94.
- **Likely reason:** the split leaf has trained on the group's images for every epoch so far, while the candidate leaf has seen few or none of them. A leaf is a full classifier over all classes, so a mixed leaf still predicts its own 4s well; a pure 4 leaf elsewhere is only about as good.

### class
- **Good:** the only policy that fired; its links on MNIST are exactly the intended "send the 0s to the pure 0 leaf" case, and they read clearly in the flow tree. Beam 1 ties `none` (0.9902 vs 0.9900).
- **Bad:** fired in only 2 of 3 MNIST seeds and never on CIFAR-10, where no leaf reaches 0.9 purity. Linked leaves can lose purity afterwards (leaf13: 98% → 84%) because they now train on a second parent's images too. Mean leaf purity is slightly lower than `none` (0.882 vs 0.892, within noise). Uses labels.

## Not tried
- **A looser `LINK_RATIO`** (e.g. 1.0: link when the existing leaf is at least as good). Ratios of 0.94–0.98 occurred, so ~1.0 would fire sometimes. Untested whether those links help.
- **A lower `LINK_PURITY` for CIFAR-10.** Best candidates were 0.23–0.43 pure, so even 0.5 would rarely fire.
- **Links under the `pure` preset** (`--split-policy class`). Class splits make purer leaves, which should give `class` links more candidates. Not run: the plan tested links on 0002's default only.
- **Linking outside split time**, and to routers: out of scope for this feature.

## Decision
**Kept: `--link-policy none` for both presets.** No code or README default changes, so no ADR. `loss` and `class` stay as flags with their tests (`docs/memory/growth-policies-stay.md`).

**Why:** links had no measurable effect. `loss` never fired at `LINK_RATIO = 0.5`; `class` fired 3 times in 9 MNIST runs and never on CIFAR-10. Beam 1 and beam 3 tied `none`, and merges still did not fire.

## Follow-up experiment (planned, not run)
Give linking a fairer chance before ruling it out. 18 runs, about 7 h, same fixed setup as above (`--max-nodes 19`, MNIST 12 / CIFAR-10 15 epochs, seeds 0–2, `--merge`):

- **loss-1.0:** `--preset explainable --link-policy loss` with `LINK_RATIO = 1.0`: link when the existing leaf is at least as good on the group as the split leaf. Ratios of 0.94–0.98 occurred here, so it should fire sometimes.
- **pure-none:** `--preset pure --link-policy none` (baseline for the next variant).
- **pure-class:** `--preset pure --link-policy class`. Class splits make purer leaves, so `class` links should find more candidates.

Needs first: a `--link-ratio` flag (today `LINK_RATIO` is a constant in `beamroot/growth.py`) and a grid in `beamroot/experiment.py`. Report links fired, beam-3 − beam-1, merges, and purity drift of linked leaves (leaf13 here went 98% → 84%). Change a preset's `link_policy` only if a policy beats `none` on beam 1 or ties it with clearly more readable graphs.
