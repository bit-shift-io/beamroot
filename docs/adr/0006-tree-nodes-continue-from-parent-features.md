# ADR 0006: Tree nodes continue from their parent's features

**Status:** Accepted
**Date:** 2026-11-01

## Context
On CIFAR-10, no routed model beat a plain network of equal MACs ([experiment 0006](../experiments/0006-cascade-cifar.md)):
- Cascade nodes (ADR 0005) read the raw image, so every node relearns the same early features on a fraction of the data.
- Splits grouped images by clustering hidden activations. Routers had no stable target and stayed undecided.

[Experiment 0007](../experiments/0007-where-to-split.md) found what works:
- a shared conv trunk;
- a router trained on class groups taken from the network's own confusion matrix;
- branches that continue from the trunk's feature maps.

At equal MACs this tied the plain network, with decisive routers. At reduced MACs it beat a no-routing network of equal cost by 2–3 points.

## Decision
- Add a third model kind, `--model tree`. Its root is a plain conv network (the `deep4` blocks).
- A split cuts a leaf's segment at a block. The router reads that block's output. Each child continues from it with a copy of the remaining blocks, narrowed to the channels its group activates.
- Splits are supervised:
  - groups: the binary top cut of the leaf's class-confusion hierarchy;
  - block: the earliest one whose routing accuracy passes a bar, or else the last block.
- Training is teacher-forced: routers learn their group labels, and leaves learn the images whose true path reaches them.

## Alternatives Considered
- **Change `dag` growth to split inside the stem:** breaks reproduction of experiments 0001–0006.
- **Cascade with a shared stem:** keeps unsupervised cluster splits, which 0006 showed stay undecided.
- **Unsupervised splits in the tree:** contradicts the 0007 evidence. Explainability needs routers tied to class groups.

## Consequences
- Trees use labels to grow. This is the supervised counterpart of the `explainable` preset's unsupervised splits, closer to the `pure` preset.
- A tree has no merges or links: one parent per node.
- Misrouted images cannot be recovered at beam-1. Early splits trade accuracy for compute.
- Beam search, stats and cost reach nodes through a model hook: a tree node's input depends on its path, not on one shared stem output.
