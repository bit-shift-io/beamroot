# Experiment 0006: Cascade on CIFAR-10

**Date:** 2026-11-01 (launched)
**Status:** seed 0 done; decision pending review

## Hypothesis
Experiments [0004](0004-cascade-sizing.md) and [0005](0005-router-freeze.md) were MNIST only. Over three seeds, `both-classifier-copy-settle` was 3.0 points below `dag` at about 1.8× fewer beam-1 MACs. CIFAR-10 is harder in exactly the way the cascade's cheap early nodes are weakest: small step-1 nodes see a downsized image, and colour and texture detail is lost at low resolution. We expect the accuracy gap to `dag` to widen on CIFAR-10, most of all on the `resolution` axis, and the MAC saving to shrink if routers near the root stay undecided and keep getting bumped.

The main criterion is beam-1 test accuracy. Cost axes are beam-1 MACs and trained node evaluations per training image; wall clock is not comparable between runs (0004). Leaf purity, split-router entropy and bumps break ties. Soft accuracy is reported, not ranked.

## Variants
Grid `cascadecifar` in `beamroot/experiment.py`. Every variant reuses an earlier grid's flag list by reference.
- **dag:** the `cascade` grid's `dag`: `--model dag --merge --max-nodes 19 --preset explainable`.
- **both-classifier-copy-settle:** the `cascadeinit` variant: `--model cascade --leaf classifier --max-nodes 19 --start-step 1 --size-axis both --leaf-init copy --settle-epochs 3 --bump-cooldown 2`. The best cost point on MNIST.
- **resolution-classifier-copy:** the `cascadeinit` variant: `--model cascade --leaf classifier --max-nodes 19 --start-step 1 --size-axis resolution --leaf-init copy`. The most accurate cascade on MNIST (0.968, one seed), and the axis expected to suffer most on CIFAR-10.
- **both-classifier-copy-settle-start4** (added after the first three finished): `both-classifier-copy-settle` with `--start-step 4`, so every node starts at the largest size. It tests the first half of the "start big, then compact" idea: does a big cascade router become decisive on CIFAR-10 when small ones do not? The first attempt ran MPS out of memory at the epoch-2 split. `_pretrain_router` fit the new step-4 router on every collected image in one batch. It now accumulates each full-batch step over 512-image chunks: same update, bounded memory, test `test_router_pretrain_runs_in_chunks_with_the_full_batch_update`. The run was restarted from scratch.

Dropped to keep the grid short: label leaves (written off in 0004), fresh-leaf variants (`copy` beat them by 2–3 points), `hidden-classifier` (2.5× the MACs of `dag`, no saving), single-knob `copy`/`settle` variants, and the freeze variants (not adopted in 0005).

## Setup
- **Command:** `.venv/bin/python -m beamroot.experiment cascadecifar --datasets cifar10 --name cascadecifar-s0`, in the background, logging to `runs/experiments/cascadecifar-s0/log.txt`. Each run is `train.run` with `--no-report` plus the variant's flags.
- **Dataset:** CIFAR-10, 15 epochs, seed 0 only (seeds 1–2 to follow for the comparisons that matter), device MPS.
- **Code:** commit `cc56dd3` plus the uncommitted router-freeze changes and this grid. From this grid on, `experiment.main` runs seeds in the outer loop and variants in the inner loop, so with several seeds a slow stretch on the machine is spread over every variant instead of landing on one (in 0005 the last variant ran last for all three seeds).
- **Results:** `runs/experiments/cascadecifar-s0/` (`summary.json`, `summary.md`, `log.txt`).
- **Probe (MPS, not evidence):** `both-classifier-copy-settle`, seed 0. With `--limit-batches 50` (which caps training batches too), 4- and 7-epoch runs exercised the split with leaf copy, bumps (root 1 → 2 → 3), growth to 11 nodes and the `measure` step without errors. Two full epochs took 41 s and 44 s (test accuracy 0.41). The root's entropy stayed at 0.67–0.69 in the limited runs, but those had seen about 100 training batches, so this says nothing about full runs; experiment 0001 saw CIFAR split routers fall from 0.63–0.66 to about 0.17 by epoch 5. Growth in short probes also depends on the total epoch count, so a probe does not predict a full run's graph.

## Results
Seed 0, one run each. MACs are beam-1 MACs per image; `node evals` is node evaluations per training image (cascade only).

| variant | beam-1 acc | soft acc | leaves | leaf purity | split-router entropy | beam-1 MACs | bumps | mean step | node evals | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| dag | **0.7314** | 0.7342 | 10 | **0.460** | **0.054** | **6.97M** | 0 | – | – | 21.4 |
| resolution-classifier-copy | 0.5386 | 0.5511 | 10 | 0.287 | 0.571 | 14.76M | 24 | 2.26 | 12.2 | 58.5 |
| both-classifier-copy-settle | 0.5070 | 0.5129 | 10 | 0.198 | 0.586 | 15.04M | 22 | 2.16 | 12.6 | 47.6 |
| both-classifier-copy-settle-start4 | **0.7351** | **0.7511** | 10 | 0.202 | 0.525 | 29.39M | 0 | 4.00 | 10.4 | 85.0 |

The start4 run was trained after the other three, alone on the machine apart from CPU-only probes, so its train minutes are not comparable. Its bumps read 0 because every node is already at the largest step: failed size checks are logged `at max`, not counted.

## Per-variant notes
- **dag:** 0.731 beam-1 at 6.97M MACs in 21 minutes. Its routers are decisive (mean split-router entropy 0.054, root 0.028), as experiment 0001 found on CIFAR-10. The root splits animals (classes 2–7) from vehicles (0, 1, 8, 9).
- **Both cascade variants lose on accuracy and on cost.** They are 19–22 points below `dag` and use about 2.1× its beam-1 MACs. On MNIST they were 3 points below at 1.8× fewer MACs. Their routers stay undecided: split-router entropy 0.57–0.59 against a maximum of ln 2 ≈ 0.693 for two children. The size check keeps failing, so 22–24 bumps push most nodes towards the largest step (mean step about 2.2, many logged `at max`). Undecided routers send each image down most branches: about 12 node evaluations per training image out of 19 nodes, against about 4.4 on MNIST. Leaves barely specialise: purity 0.20–0.29, against 0.10 for leaves that sort images no better than chance.
- **Resolution vs both:** `resolution-classifier-copy` is 3 points better than `both-classifier-copy-settle` (0.539 against 0.507), the reverse of what the hypothesis expected, but both are far behind `dag` and the gap between them is inside what one seed can show.
- **Likely reason, unverified:** a cascade router owns its stem, and only the routing gradient trains it. A `dag` router reads the shared stem, which every leaf's classification loss trains, so it routes on features already shaped by the classes. Small cascade routers cannot find a split on CIFAR-10. Bumps restart them fresh at a larger size, which costs MACs, and the restarts cost the leaves below them training time. Probes in [Compaction probes](#compaction-probes) test whether the size or the stem training is what is missing.

## Compaction probes
The idea: start each node big, train until it splits its images into a few decisive groups, then compact it into a much smaller model that reproduces the split. These probes test the compaction step separately from the growth step. The scripts live outside the repo (session scratchpad); none of this is in `beamroot/` yet.

### P2: can small cascade routers imitate the dag's decisive routers?
- **Setup:** the teacher is the `dag` run above (beam-1 0.7314). For each of its 9 routers, students are cascade `NodeModel` routers (their own stem, raw 32×32 image in) at steps 1–3 on the `both` and `resolution` axes, plus step 4. They are trained for 8 epochs with KL(teacher ‖ student), weighted by how much of each image's probability reaches that router (teacher reach). Augmentation is crop plus flip. The kept checkpoint is the one with the best validation agreement. MPS, 70 min.
- **One router swapped at a time:** argmax agreement with the teacher is 0.77–0.96, and it rises with student size: the root reaches 0.92 at step 1 and 0.96 at step 4; `leaf7`, the hardest router, only 0.77–0.89. Beam-1 accuracy barely moves whatever the size: 0.7214–0.7316 against 0.7314. The images students route differently are mostly images the leaves would get wrong anyway.
- **Every router swapped at once (same config everywhere):**

| student routers | beam-1 acc | routing MACs/image |
|---|---|---|
| teacher (dag heads) | 0.7314 | (shared stem) |
| both:1 | 0.7022 | 0.12M |
| resolution:1 | 0.7038 | 0.16M |
| both:2 | 0.7177 | 1.39M |
| resolution:2 | 0.7166 | 1.43M |
| both:3 | 0.7217 | 6.26M |
| resolution:3 | 0.7249 | 6.33M |
| both:4 | 0.7269 | 18.7M |

- **Choosing students by an agreement bar does not work well.** Picking, per router, the cheapest student whose validation agreement is at least 0.90 gives 0.7206 at 3.24M. A bar of 0.95 or higher keeps nearly every teacher. Agreement says less about the cost to accuracy than the whole-graph swap does.
- **Finding:** the dag's decisive routing can be compacted. Step-1 routers, at 36K MACs each and about 3.3 on a beam-1 path, keep beam-1 accuracy within 2.9 points; step-2 routers keep it within 1.4 points. This alone saves nothing in the dag, though. Its routers are cheap heads (262K each) on a shared stem that costs 5.6M of its 6.97M MACs, and the leaves need that stem anyway. The saving only appears if the leaves can also run without the shared stem: see P3.

### P3: can the leaves be small too, once compact routers narrow their job?
- **Setup:** route every image beam-1 through the P2 student routers (all `both:1` or all `both:2`). For each of the dag's 10 leaves, train cascade classifier leaves (`NodeModel`, steps 1–3 on the `both` axis) only on the training images routed to that leaf. The loss is 0.5 × cross-entropy plus 0.5 × KL(dag leaf ‖ student). Augmentation is crop plus flip, 15 epochs, Adam 1e-3, keeping the best validation epoch, on CPU.
- **Control (`flat`):** a single cascade classifier of the same sizes, plus step 4, with no routing. It sees every image and is distilled from the whole dag's output. If routing earns its keep, a routed pipeline should beat a flat model of equal MACs.

| pipeline | test acc | MACs/image |
|---|---|---|
| dag (teacher) | 0.7314 | 6.97M |
| flat step 1 | 0.4769 | 0.04M |
| both:1 routers + step-1 leaves | 0.4879 | 0.16M |
| flat step 2 | **0.6111** | 0.44M |
| both:1 routers + step-2 leaves | 0.5964 | 0.56M |
| both:2 routers + step-2 leaves | 0.6140 | 1.83M |
| flat step 3 | **0.6782** | 2.03M |
| both:1 routers + step-3 leaves | 0.6448 | 2.15M |
| both:2 routers + step-3 leaves | 0.6559 | 3.42M |
| flat step 4 | **0.7394** | 6.13M |

- **Routing does not pay for itself here.** At every leaf size, the flat model matches or beats the routed pipeline for fewer MACs: 3.3 points better at step 3, 1.5 at step 2. Only at step 1, where every model is poor, does routing add 1 point, and it costs 4× the MACs. A routed leaf trains on about a tenth of the data, and the routes do not narrow the job enough to make up for that. The dag's beam-1 leaves are not pure: `leaf4` gets the four vehicle classes, but `leaf17` and `leaf15` get six or more classes each (test class mix in the probe output).
- **A flat step-4 cascade node distilled from the dag beats the dag:** 0.7394 against 0.7314, at 6.13M against 6.97M MACs. Both trained with the same crop-and-flip augmentation.
- **Distillation is not why the flat model wins.** The same flat models trained on hard labels only (`--kd 0`, no teacher) reached 0.6223 at step 2, 0.6742 at step 3 and **0.7423** at step 4, within noise of the distilled ones. A single standalone network of the dag's size, trained for the same 15 epochs, matches or beats the whole routed dag on CIFAR-10 (0.742 against 0.731, one seed each). Its advantages are that it trains one fixed network for all 15 epochs, with no epochs spent growing, and it keeps the best-validation epoch. One seed, and all validation curves were still rising at epoch 15.
- **Finding:** on CIFAR-10 with these node sizes, compacting the dag into a routed cascade of small nodes is accurate only when nodes are large enough that a single flat node of that size is better still. Compaction works for routers (P2) but not for the system as a whole. The more basic result is the flat baseline: none of the routed models so far beats one plain network of equal MACs on CIFAR-10. The experiment log has never run this baseline as a grid variant.

### P4: does a cascade that starts big learn decisive routers? (`-start4` in Results)
- **Accuracy: yes, level with the dag.** Beam-1 0.7351 against 0.7314; soft accuracy 0.7511 is the best of the grid. Starting big fixes the cascade's 20-point accuracy gap. Test accuracy was 0.60 after 2 epochs and 0.72 after 8, so it learns fast.
- **Decisive: mostly no.** Mean split-router entropy is 0.525, against 0.054 for the dag. Only the root (0.20) and `leaf1` (0.24) are decisive. The other seven routers sit at 0.41–0.69, and 0.69 is a coin flip. Leaf purity is 0.20, as low as the small cascades.
- **Beam-1 uses half the graph.** Five routers' argmax always picks the same child. On the test set, beam-1 sends images to only 5 of the 10 leaves (1510, 2379, 2671, 3413 and 27 images), with purity 0.19–0.28. Each of those leaves is a step-4 classifier over all 10 classes, the same size as the flat step-4 model in P3. In beam-1 terms the run is about four copies of the flat network behind a soft, roughly vehicles-versus-animals split. The other five leaves only contribute to soft accuracy.
- **Cost:** 29.4M beam-1 MACs, 4.2× the dag. Every node on the path, routers included, is a full step-4 network.

### P5: can the start4 cascade's routers be compacted?
- **Setup:** as P2, with the start4 run as the teacher (MPS, 66 min).
- **Yes, almost for free.** With every router swapped for a step-1 student (36K MACs each), beam-1 is 0.7317 against 0.7351. Routing MACs per image fall from 23.3M to 0.145M. The root's step-1 student agrees with it on 96% of images. Routers whose argmax never changes agree 100% at any size. The cheapest student per router at a 0.95 agreement bar gives 0.7326 at 0.40M routing MACs.
- **Compacted start4 against the alternatives:** routing 0.145M plus one step-4 leaf, 6.13M, gives about 6.27M MACs at 0.7317. That ties the dag (0.7314 at 6.97M) for 10% fewer MACs. It loses to the flat step-4 network (0.742 at 6.13M), which is simpler and has no routing. Its leaves are no purer than the small cascades', so it is not more explainable either.

### What the probes say about "start big, then compact"
- **The compaction half works.** Big routers, from the dag (P2) or from a start4 cascade (P5), can be replaced by step-1 students for 0.3–3 points of beam-1 accuracy, at about 1% of their MACs.
- **The "learn enough to divide into a few groups" half does not happen on CIFAR-10 at this scale.** A big cascade router does become accurate, but only the top one or two routers become decisive. The leaves stay 10-class generalists, and beam-1 ignores half of them. The dag's routers are decisive (P2) but its leaves are still not pure (0.46).
- **No routed model here beats a flat network of equal MACs (P3).** Cheap routing is only worth having if it lets the leaves be small, and on CIFAR-10 small leaves are worse than one small flat network.

## Not tried
- **More seeds.** Seed 0 first; CIFAR-10 seed spread can exceed variant spread (0001).
- **A looser bump bar near the root, or stopping bumps that do not improve a router.** Proposed if full runs show roots that stay undecided past epoch 5 and keep bumping. The full runs did show this, but start4 removes bumps entirely and its routers still stay undecided, so a looser bar would only stop the MACs growing. It would not make the routers decide.
- **A flat-network variant in the grid** (`--model` with a single classifier node, at the dag's MACs, several seeds). P3 suggests it beats every routed model on CIFAR-10. It should be the reference every future experiment has to beat on accuracy per MAC. It is not yet run under `train.py`.
- **Compaction built into growth:** after a big router passes the size check, distil it into the smallest student that keeps validation accuracy, and bypass routers whose argmax never changes. P5 suggests this would cut start4's MACs to about the dag's level. Not done, because it would still lose to the flat network.
- **Training that pushes routers to decide** (an entropy penalty, or hard routing in the later epochs), so leaves get a narrow class set and can be small. Leaf purity, not router size, is what blocks small leaves (P3).
- **Longer training.** Every run and probe curve was still rising at epoch 15.

## Decision
Pending the user's review. The evidence so far:
- The small-start cascade loses on CIFAR-10: 19–22 points below the dag, at 2.1× its MACs.
- The start-big cascade ties the dag on accuracy at 4.2× the MACs, or at 0.9× once its routers are compacted. Its leaves are no purer.
- A flat network beats both (one seed).

Seeds 1–2 for `dag`, `-start4` and a flat baseline would be needed before adopting anything.
