# Experiment 0007: Where can a CIFAR-10 network be split?

**Date:** 2026-11-01
**Status:** pending user review

## Hypothesis
Experiment [0006](0006-cascade-cifar.md) found that no routed model beat one plain network of equal MACs on CIFAR-10. Cascade routers and small leaves that read the raw image stayed undecided and weak. The likely reason is that early features (edges, colours, textures) are shared by every class, so splitting at the input makes every node relearn them on a fraction of the data. A network should split where its classes have become separable, with children reading the parent's features rather than the image, and on class groups the network itself finds hard to tell apart.

The probes ask four questions:
1. **Depth:** at which depth of a plain network does a decisive class-group split become possible?
2. **Tree (ideas 1 + 2):** does a shared trunk, a supervised group router and per-group branches that continue from the trunk's feature maps beat a plain network of equal MACs?
3. **Tree at the last layer (idea 3):** what does a class-hierarchy decision on the final features cost in accuracy?
4. **Early exit (idea 4):** how much compute does answering easy images at an early layer save?

Main criterion: test accuracy. Cost: MACs per image. Ties go to the more explainable model (decisive routers, pure branches).

## Setup
- **Scripts:** off-repo probes in the session scratchpad (`depth_probe.py`, `tree_probe.py`, sharing helpers with the 0006 probes). Nothing in `beamroot/` is used except `build_stem`, `module_macs` and the data split.
- **Data:** CIFAR-10, the project's train/val split (seed 0), crop-and-flip augmentation, 15 epochs, Adam 1e-3, best-validation epoch kept, MPS. One seed throughout.
- **Plain networks:**
  - `stem2`: the project's shared stem (2 conv blocks) plus a 128-unit hidden layer. The same as the flat step-4 node in 0006 P3, 6.13M MACs.
  - `deep4`: 4 conv blocks (16, 32, 64, 128 channels; two convs in the last), global average pool, linear, 6.34M MACs. It has more depths to probe at the same cost.
- **Class hierarchy:** average-linkage clustering of the classes on 1 − the symmetric validation confusion rate of the trained plain network, cut into 2 and 4 groups.
- **Depth heads:** each network is frozen. At every probe point, a small head (average-pool to at most 4×4, Linear 64, ReLU, Linear) is trained for 8 epochs per task: the 2-group cut, the 4-group cut, and 10 classes.

## Results

### 1. Depth: where does a decisive split become possible?
Both networks found the same hierarchy:
- vehicles {plane, car, ship, truck} against animals;
- then {plane, ship}, {car, truck}, {bird, cat, deer, dog, horse} and {frog} (the 4-group cut);
- deeper merges: cat+dog first (the most confused pair), then deer+horse, then bird.

Test accuracy and mean entropy of each head (entropy maximum: ln 2 = 0.69 for 2 groups, ln 4 = 1.39 for 4):

| net | probe point | trunk MACs | 2-group acc | entropy | 4-group acc | entropy | 10-class acc |
|---|---|---|---|---|---|---|---|
| deep4 | pixels | 0 | 0.858 | 0.37 | 0.711 | 0.77 | 0.432 |
| deep4 | b1 | 0.44M | 0.914 | 0.22 | 0.812 | 0.56 | 0.554 |
| deep4 | b2 | 1.62M | 0.928 | 0.19 | 0.850 | 0.43 | 0.655 |
| deep4 | b3 | 2.80M | 0.942 | 0.15 | 0.883 | 0.32 | 0.727 |
| deep4 | b4 | 6.34M | 0.952 | 0.11 | 0.894 | 0.25 | 0.735 |
| stem2 | pixels | 0 | 0.856 | 0.37 | 0.710 | 0.77 | 0.426 |
| stem2 | b1 | 0.88M | 0.912 | 0.21 | 0.822 | 0.50 | 0.599 |
| stem2 | b2 (the dag's stem) | 5.60M | 0.942 | 0.15 | 0.875 | 0.32 | 0.730 |
| stem2 | hidden | 6.13M | 0.953 | 0.12 | 0.895 | 0.26 | 0.746 |

The plain networks themselves reach 0.7306 (`deep4`) and 0.7379 (`stem2`).

- **Vehicles against animals is decisive from the first block.** After one conv block (0.44M MACs) a router gets it right 91% of the time, at entropy 0.22. Even 4×4-pooled raw pixels get 86%: colour and background (sky, water, road against grass) carry most of it. The 0006 cascades never found this split because their routers were trained by clustering, not on class groups.
- **But routing errors only fall slowly with depth:** 8.6% at b1, 5.8% at b3, 4.8% at the last layer. Every misrouted image is lost if a branch only knows its group's classes, so an early split caps accuracy unless the branches are better than the plain network's remainder.
- **The 4-group cut needs depth:** 81% at b1, 89% at the end.

### 4. Early exit
Exit heads on the frozen network: an image exits at the first probe point where the 10-class head's top probability is at least the threshold; the rest go through the whole network. MACs include every exit head evaluated.

| net | threshold | acc | mean MACs | exited early |
|---|---|---|---|---|
| deep4 | none (plain) | 0.7306 | 6.34M | 0% |
| deep4 | 0.95 | 0.7308 | 5.61M (−12%) | 33% |
| deep4 | 0.9 | 0.7301 | 5.13M (−19%) | 44% |
| deep4 | 0.8 | 0.7281 | 4.34M (−32%) | 59% |
| deep4 | 0.7 | 0.7232 | 3.60M (−43%) | 72% |
| stem2 | none (plain) | 0.7379 | 6.13M | 0% |
| stem2 | 0.9 | 0.7373 | 5.73M (−7%) | 28% |
| stem2 | 0.8 | 0.7347 | 5.20M (−15%) | 44% |

- Early exit works on a deep, evenly costed network: `deep4` saves 19% at no loss and 32% for 0.25 points. It works poorly on `stem2`, whose second block is 85% of its MACs, so there is no cheap early point to exit from. These heads were bolted onto a frozen network. Training the exits jointly usually does better, but that is not tested here.
- Early exit gives no explanation. The path is "easy" or "hard", not a class group.

### 2. Tree: shared trunk, supervised router, per-group branches
- **Model:** `deep4`'s blocks up to the split point form a shared trunk. A router head (as in the depth probe) predicts the class group (2- or 4-group cut). Each group gets a branch: the rest of `deep4` at width 1.0 or 0.5, reading the trunk's feature maps and classifying only that group's classes.
- **Training:** end to end and teacher-forced. Each image trains the router on its true group and trains only its true group's branch.
- **Test:** beam-1. The router's argmax picks the one branch that runs.
- **Labels:** a config is `split:groups:width`. Configs with 1 group are no-routing controls: the same trunk plus one branch for all 10 classes.

| config | test acc | route acc | route entropy | beam-1 MACs |
|---|---|---|---|---|
| plain `deep4` | 0.7306 | – | – | 6.34M |
| b1:2:1.0 | 0.7215 | 0.915 | 0.22 | 6.36M |
| b2:2:1.0 | 0.7288 | 0.931 | 0.19 | 6.37M |
| b3:2:1.0 | 0.7302 | 0.941 | 0.15 | 6.41M |
| b3:4:1.0 | 0.7094 | 0.871 | 0.37 | 6.41M |
| b4:2:1.0 | 0.7320 | 0.953 | 0.10 | 6.37M |
| b4:4:1.0 | **0.7327** | 0.897 | 0.26 | 6.37M |
| b1:2:0.5 | 0.6666 | 0.919 | 0.21 | 2.23M |
| b1:1:0.5 (control) | 0.6410 | – | – | 2.23M |
| b2:2:0.5 | 0.6944 | 0.930 | 0.17 | 3.13M |
| b2:4:0.5 | 0.6996 | 0.855 | 0.41 | 3.13M |
| b2:1:0.5 (control) | 0.6722 | – | – | 3.13M |

- **At equal MACs, a tree that splits at b2 or later costs nothing.** b2, b3 and b4 trees are within 0.2 points of the plain network, one seed, so a tie. Their routers are decisive (entropy 0.10–0.19) and each branch only ever sees and answers its own class group. Splitting after the first block costs 0.9 points: its 8.5% routing errors are images the branches cannot answer.
- **The router learns the group better as part of the tree than as a head on a frozen network.** b4:2 routes 95.3% correctly against 95.2% for the frozen head, and b2:2 93.1% against 92.8%. The difference is small.
- **4 groups split early lose:** b3:4 is 2 points below b3:2. The 4-group cut is only 87% routable at b3, and misroutes are unrecoverable.
- **When branches are cheaper, routing pays.** With half-width branches, every tree beats the same trunk plus one half-width branch without routing:
  - 0.6666 against 0.6410 at b1, +2.6 points at 2.23M MACs;
  - 0.6944 (2 groups) and 0.6996 (4 groups) against 0.6722 at b2, +2.2 and +2.7 points at 3.13M MACs.

  The tree runs the same MACs per image but holds 2–4 branches' worth of parameters, each specialised on a few classes. That is the trade conditional computation promises: more capacity at equal compute. It is the first result in this log where routing beats a no-routing model of equal MACs on CIFAR-10. One seed, but the gain is consistent across all three configs.
- **Early exit is still cheaper at the same accuracy.** `deep4` with exits at threshold 0.6 is 0.7107 at 2.83M MACs, above the b2:4:0.5 tree (0.6996 at 3.13M). Early exit keeps the full-width network for hard images, while the tree's branches are narrow for every image. The two are not exclusive: a tree's branches could have exits too, which is not tested.

### 3. Tree decision at the last layer
`b4:2:1.0` and `b4:4:1.0` are the explainable-decision version: the whole network is shared, and the "branches" are just one linear layer per group on the final features. Beam-1 accuracy is 0.7320 and 0.7327, against 0.7306 plain. Each prediction comes with a group decision such as "{car, truck} → truck", at no cost in accuracy or MACs. Stacking both cuts would give a deeper path, "vehicle → {car, truck} → truck", but that is not tested. This is the cheapest way to get an explanation. It saves no compute, and its explanation covers only the coarse groups, not which features led there.

## Summary
- **A CIFAR-10 network can be split into a graph, but at a layer, not at the input.** Children must read the parent's feature maps. The groups must come from the class hierarchy the network finds hard to separate, not from unsupervised clusters.
- **Split at b2 or later, at equal MACs:** a tie with the plain network, with decisive routers and branches that each answer only a few classes. This is the explainability win.
- **Split with narrower branches:** 2–3 points better than the same cheap network without routing. This is the compute win, though early exit on the plain network still reaches the same accuracy for fewer MACs.
- **What beamroot's growing graph does differently, and should stop doing:** cascade nodes read the raw image (0006), and splits come from clustering hidden activations, with routers trained only through the routing loss.

## Not tried
- **More seeds.** Every number here is one seed. The tree-versus-control gains (+2.2 to +2.7) and the equal-MAC ties need seeds 1–2 before they are trusted.
- **Two-level trees** (vehicles → {car, truck} → class), with each level splitting at its own depth: the top split early, the finer splits later.
- **Growing the tree** in beamroot's style: start with the plain trunk, and split a node when its confusion matrix shows a group structure, with branches initialised by copying the parent's remaining blocks. This is what the probes suggest the growth step should become.
- **Recovering misroutes:** soft routing or beam-2 at test time, or branches that also see a little of the other group, so early splits do not lose 8% of images outright.
- **Early exit inside tree branches**, and exits trained jointly rather than bolted on.
- **Width other than 0.5**, to find where the tree's advantage over the no-routing control is largest.

## Decision
Pending the user's review.
