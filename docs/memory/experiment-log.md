---
name: experiment-log
description: Comparison runs of training or growth variants must be written up in docs/experiments/, including the losers, before the losing code is deleted
metadata:
  type: convention
---

**Why:** Losing variants get deleted from the code once a winner is adopted. Without the log, the next experiment repeats ideas already tried and loses the reasons they failed.
**How to apply:** When you run or plan a comparison of variants (losses, growth rules, initialisations, thresholds).

- Write one `docs/experiments/NNNN-*.md` per experiment, from the template in `docs/experiments/README.md`.
- Record every variant's strengths, weaknesses and likely reason, plus ideas left untried.
- Judge variants on beam-1 test accuracy first. Soft accuracy is reported, not ranked.
- With few seeds, call a gap a tie unless it is statistically meaningful (e.g. a t-test), not merely past one std. On a tie, prefer the more explainable graph (shorter beam-1 paths, purer leaves).
- Variants must differ only in the thing being compared. Code that draws from the global RNG in some variants only (e.g. iterating an extra data loader) shifts every later random draw and fakes a difference. Wrap such work in `torch.random.fork_rng`, and check that a variant whose feature never triggers matches the baseline exactly.
- Structural decisions (merges, thresholds) read the validation set, never the test set.
- Start a new grid with seed 0 only, then decide more seeds from the results. MNIST seeds agree closely; CIFAR-10 often needs 2–3, since its seed spread can exceed the variant spread (experiment 0001).
