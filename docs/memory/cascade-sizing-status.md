---
name: cascade-sizing-status
description: State of the cascade-sizing work as of 2026-10-31 — what is built, what experiment 0004 found, what is undecided, and the planned next steps; read when touching --model cascade or planning the next cascade experiment
metadata:
  type: context
---

**Why:** The work spans many sessions and long runs. Without this, a new session re-derives decisions and re-runs hours of experiments.
**How to apply:** When you touch `beamroot/models/cascade.py`, cascade growth in `beamroot/growth.py`, or the `cascade`, `cascade29` or `cascadeinit` experiment grids. Delete this entry once the decisions below are made and recorded in an ADR.

State on 2026-10-31 (code committed through `9665737`; experiment [0004](../experiments/0004-cascade-sizing.md), status pending user review):

- **Built:** `--model cascade` with sized node models (ADR 0005), growth by cluster splits, size checks and bumps, pruned soft routing, `--leaf {classifier,label}`, `evaluate` MACs and ms per image, cascade-aware report and class flow, the grids above, and the flags `--leaf-init {fresh,copy}`, `--settle-epochs`, `--bump-cooldown`. Defaults reproduce earlier behaviour (not diffed bit-for-bit).
- **Findings (MNIST, seed 0 only):** `dag` is 0.991 at 4.76M beam-1 MACs. Label leaves lost badly (0.59–0.77 at 19 nodes, 0.68–0.82 at 29 nodes, stuck near 13 leaves). `hidden`-axis cascade matches `dag` accuracy at 2.5× the MACs. `--leaf-init copy` adds about 2.5 points to the cheap `resolution` and `both` variants, and settle 3 plus cooldown 2 cuts bumps (18 to 12 or fewer) and more than halves MACs. Best cost point: `both-classifier-copy-settle`, 0.958 at 1.35M MACs (3.5× cheaper than `dag`, 3.3 points less accurate).
- **User decisions so far:** CIFAR-10 is skipped for now. Label leaves should be dropped (they are poor); their code and `--leaf label` flag stay until the user says delete, and the loss is written up in entry 0004. Everything else is undecided.
- **Not yet done:** seeds 1 and 2 for `dag` and `both-classifier-copy-settle` (the planned next run); `copy`/`settle` on the `hidden` axis; separating settle from cooldown; the unexplained rise in training time with `copy`/`settle` (46–88 min against about 30); the finish steps of the feature plan (end-to-end verification and retrospective).
- **Design points to remember:** a step-1 node pools once, not twice, at resolution 8 or below (`SMALL_RES`), otherwise it has 16 features; label-leaf logits start at N(0, 0.5²) because uniform leaves give routers zero gradient; cascade variants must pass `--start-step 1` or no size check can ever bump (default step 4 is the maximum).
