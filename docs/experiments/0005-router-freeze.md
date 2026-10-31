# Experiment 0005: Router freeze and router decay

**Date:** 2026-10-31 (launched), 2026-11-01 (results)
**Status:** done; freeze and decay not adopted

## Hypothesis
In the cascade model an ancestor router keeps training on every image that reaches it, even after its split is settled, and keeps drifting while the nodes below it adapt to its routing. Experiment [0004](0004-cascade-sizing.md) left `both-classifier-copy-settle` as the best cost point (0.958 beam-1 at 1.35M MACs, one seed), 3.3 points below `dag`. We expect that freezing decisive ancestor routers (**Router freeze**) cuts the trained node evaluations per image without losing beam-1 accuracy, and may gain some because the leaves train against a fixed routing. Decaying ancestor learning rates first (**Router decay**) should let routers settle before they are judged, so more of them freeze. The grid also gives seeds 1 and 2 for the 0004 headline comparison (`dag` vs `both-classifier-copy-settle`).

The main criterion is beam-1 test accuracy. Cost axes are beam-1 MACs and trained node evaluations per training image (`trained node evals/image`); wall clock is not comparable between runs on this machine (see 0004). Frozen-router count, leaf purity and bumps break ties. Soft accuracy is reported, not ranked.

## Variants
Grid `cascadefreeze` in `beamroot/experiment.py`. The first two variants reuse the 0004 flag lists by reference, so they cannot drift.
- **dag:** the `cascade` grid's `dag` variant: `--model dag --merge --max-nodes 19 --preset explainable`.
- **both-classifier-copy-settle:** the `cascadeinit` grid's variant: `--model cascade --leaf classifier --max-nodes 19 --start-step 1 --size-axis both --leaf-init copy --settle-epochs 3 --bump-cooldown 2`.
- **both-classifier-copy-settle-freeze:** plus `--freeze-routers quality`. An ancestor router whose entropy is at most `--freeze-entropy-max` (0.25) × ln k and whose smallest usage is at least `--spread-min` (0.1) × 2/k stops training for good. It still routes and skips size checks. Checked after every epoch except the last, settle epochs included.
- **both-classifier-copy-settle-decay-freeze:** plus `--router-lr-decay 0.5 --freeze-routers quality`. An ancestor router's learning rate halves each epoch after the first split below it; a bump restores the full rate.

## Setup
- **Command:** `.venv/bin/python -m beamroot.experiment cascadefreeze --datasets mnist --name cascadefreeze-mnist`, in the background, logging to `runs/experiments/cascadefreeze-mnist/log.txt`. Each run is `train.run` with `--no-report` plus the variant's flags.
- **Dataset:** MNIST, 12 epochs, seeds 0, 1, 2, device MPS. Fixed flags as 0004.
- **Code:** commit `cc56dd3` plus the uncommitted router-freeze changes (slices 01–04: per-node Adam groups, `--router-lr-decay`, `--freeze-routers`, `trained_node_evals`, this grid).
- **Copied runs:** the seed-0 `dag` run (`runs/experiments/cascade-mnist-s0/20261030-145752-mnist-seed0-dag`) and the seed-0 `both-classifier-copy-settle` run (`runs/experiments/cascadeinit-mnist-s0/20261031-144543-mnist-seed0-both-classifier-copy-settle`) were copied into `runs/experiments/cascadefreeze-mnist/`, where `finished_run` picks them up and skips them, so 10 runs train. Both predate the router-freeze changes. They are comparable because slice 01's test `test_node_param_groups_train_like_one_group` shows that per-node Adam groups at the full rate train exactly like the old single group, and with both flags off nothing else changes (no extra RNG draws). The `dag` model is untouched by those changes. The copied runs have no `trained_node_evals` or `freezes` in their metrics, and `summarise` drops missing values, so the `both-classifier-copy-settle` row's `trained node evals/image` and `frozen routers` average seeds 1 and 2 only (with no freeze flag, trained equals `train node evals/image` anyway). Read seed-0 values from `summary.json` per run when comparing.
- **Results:** `runs/experiments/cascadefreeze-mnist/` (`summary.json`, `summary.md`, `log.txt`). Wall clock 11 h 1 min for the 10 runs (2026-10-31 21:40 to 2026-11-01 08:41), alone on the machine.
- **Smoke (MPS, not evidence):** `-decay-freeze`, seed 0, `--epochs 6 --limit-batches 100`, default bars. `root` froze after epoch 3 (entropy 0.162, spread 0.370); `trained_node_evals` was `node_evals` − 1 from epoch 4 on (3.64 against 4.64). A 2-epoch smoke cannot freeze anything with `--settle-epochs 3`, because no split happens, so no router becomes an ancestor. Note that `train --limit-batches` caps training batches as well as test batches.

## Results
Mean ± std over seeds 0–2. MACs are beam-1 MACs per image. `trained evals` is trained node evaluations per training image; for the baseline it equals `node evals` (nothing freezes), so seed 0 is read from its `train node evals/image`.

| variant | beam-1 acc | leaves | leaf purity | beam-1 MACs | bumps | node evals | trained evals | frozen routers | train min |
|---|---|---|---|---|---|---|---|---|---|
| dag | 0.9900 ± 0.0009 | 12.0 | 0.892 ± 0.024 | 4.74M | 0 | – | – | – | 18.2 |
| both-classifier-copy-settle | 0.9602 ± 0.0034 | 11.0 | 0.776 ± 0.034 | 2.68M ± 1.21M | 10.7 | 4.36 ± 0.37 | 4.36 ± 0.37 | 0 | 61.3 |
| -freeze | 0.9560 ± 0.0043 | 11.0 | 0.675 ± 0.046 | 2.63M ± 1.33M | 10.7 | 4.55 ± 0.47 | 3.50 ± 0.24 | 1.3 | 65.6 |
| -decay-freeze | 0.9559 ± 0.0042 | 11.3 | 0.746 ± 0.034 | 2.67M ± 1.33M | 10.3 | 4.69 ± 0.54 | 3.65 ± 0.24 | 1.3 | 94.4 |

Paired by seed against `both-classifier-copy-settle` (seed 0, 1, 2; paired t with 2 degrees of freedom, two-tailed 5% critical value 4.30):

| comparison | per-seed difference | mean | t |
|---|---|---|---|
| -freeze beam-1 acc | −0.0068, −0.0059, +0.0002 | −0.0042 | −1.9 (tie) |
| -decay-freeze beam-1 acc | −0.0067, −0.0059, −0.0002 | −0.0043 | −2.1 (tie) |
| -freeze leaf purity | −0.086, −0.111, −0.108 | −0.102 | −13.0 (significant) |
| -decay-freeze leaf purity | −0.046, −0.066, +0.019 | −0.031 | −1.2 (tie) |
| -decay-freeze vs -freeze leaf purity | +0.041, +0.045, +0.127 | +0.071 | +2.5 (tie) |
| -freeze trained evals | −0.75, −1.20, −0.64 | −0.86 | |
| dag − copy-settle beam-1 acc | +0.0332, +0.0257, +0.0305 | +0.0298 | +13.6 (significant) |

Freeze events (all after the first split, during settle epochs): `root` froze in every freeze run (entropy 0.04–0.07, spread 0.23–0.48), and `leaf1` also froze in seed 1 of both variants (entropy 0.15, spread 0.45–0.47).

## Per-variant notes
- **dag:** 0.990 ± 0.001 over three seeds, at 4.74M MACs and 18 minutes. Seeds agree closely, as expected on MNIST.
- **both-classifier-copy-settle (the 0004 headline, now three seeds):** 0.960 ± 0.003, 3.0 points below `dag`, and the gap is significant (t = 13.6). Its cost varies a lot between seeds: beam-1 MACs 1.35M, 3.72M and 2.96M, because the grown graph differs (10–12 leaves, 8–12 bumps). Seed 0's 1.35M, which 0004 reported as 3.5× cheaper than `dag`, was the cheapest of the three; the mean is 2.68M, about 1.8× cheaper.
- **-freeze:** works as built. Every run froze `root`, and trained node evals per image fell by 0.86 on average (about 20%). Accuracy is a tie with the baseline (−0.4 points, t = −1.9): seeds 0 and 1 lost about 0.6 points and seed 2 nothing. Leaf purity fell by 0.10 in every seed (t = −13), the one significant effect of the grid, so the graphs are clearly less explainable. Likely reason, unverified: the root freezes as soon as it is decisive, early in settle, and its routing is then fixed while the subtree below is still settling, so leaves cannot pull it towards a cleaner class split. Total node evals per image rose slightly (4.36 → 4.55), so part of the saving is given back by images visiting more nodes; why is unknown.
- **-decay-freeze:** the same accuracy as `-freeze` (tie with the baseline, −0.4 points) and the same freezes, but most of the purity loss is recovered (−0.03 against the baseline, a tie; +0.07 over `-freeze`, not significant with three seeds). Halving the ancestors' learning rate each epoch before they freeze seems to let them move less abruptly while the subtree trains. Trained node evals 3.65, about 16% below the baseline.
- **Training time:** `train min` rose (61 → 66 → 94) rather than falling. Wall clock is not comparable between runs on this machine (0004), and the saved backward pass of one or two routers is a small part of an epoch whose time is dominated by the leaves and the growth step. The deterministic saving is the trained-evals column, not the wall clock.

## Not tried
- **Freeze without settle, or decay without freeze.** The grid isolates freeze on the best 0004 variant and adds decay only on top of freeze, to stay at 10 overnight runs.
- **Other freeze bars or decay rates.** Only the defaults (`--freeze-entropy-max 0.25`, `--spread-min 0.1`) and D = 0.5.
- **Freezing later.** A minimum router age, or no freezing during settle epochs, could avoid fixing the root before its subtree settles. That is the likely cause of the purity loss.
- **Decay alone.** Decay recovered most of the purity loss that freeze caused, so decay without freeze may be worth a run on its own.
- **Why total node evals rise under freeze** (4.36 → 4.55 per image) was not investigated.
- **CIFAR-10**, as in 0004 (skipped by the user's decision).

## Decision
Neither variant is adopted. Both flags stay, off by default, so these runs stay reproducible.
- **-freeze** is a tie on beam-1 accuracy but significantly less pure (−0.10 leaf purity in every seed). Ties go to the more explainable graph, so the baseline wins.
- **-decay-freeze** is a tie on both accuracy and purity. It saves about 16% of trained node evaluations, but with an accuracy cost that is consistent in two seeds out of three and no wall-clock saving, the saving is not worth a default change on this evidence.
- The 0004 headline holds with three seeds: `both-classifier-copy-settle` is 3.0 points below `dag` (significant), but its MAC saving is about 1.8×, not the 3.5× of seed 0.
