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
- Structural decisions (merges, thresholds) read the validation set, never the test set.
