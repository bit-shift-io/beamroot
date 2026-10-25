---
name: growth-policies-stay
description: Split and link policies stay as selectable flags after their benchmarks; experiments only change the default
metadata:
  type: convention
---

**Why:** The user wants to choose between unsupervised (`cluster`, `loss`) and label-driven (`class`) growth per run. This differs from ADR 0003, where losing modes were deleted.
**How to apply:** When adopting an experiment winner for `--split-policy`, `--link-policy` or `--max-k`.

- Change the default and the README. Keep the losing policy's code, flag and tests.
- Record the evidence in `docs/experiments/` as usual.
