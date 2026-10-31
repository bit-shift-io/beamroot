# Repo memory

Situational conventions and in-flight context. Open an entry when its hook touches what you're doing.

- [Torch-pinned environment](torch-pinned-environment.md) — new deps can pull numpy 2.x and break torch 2.2.2; tests run on CPU and miss MPS-only bugs (float64, index_copy, adaptive pool); smoke on MPS yourself
- [Experiment log](experiment-log.md) — comparing training or growth variants; write up losers before deleting them; rank by beam-1, ties by significance; isolate RNG between variants; start with seed 0 only; `--limit-batches` caps only test batches, so grids take hours; probe one node before a long grid
- [Growth policies stay](growth-policies-stay.md) — adopting a split/link policy winner; change the default, never delete the loser
- [Cascade sizing status](cascade-sizing-status.md) — 2026-10-31 state of the cascade work: what is built, experiment 0004 findings, undecided points, next steps
