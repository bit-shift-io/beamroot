# Repo memory

Situational conventions and in-flight context. Open an entry when its hook touches what you're doing.

- [Torch-pinned environment](torch-pinned-environment.md) — new deps can pull numpy 2.x and break torch 2.2.2; tests run on CPU and miss MPS-only bugs
- [Experiment log](experiment-log.md) — comparing training or growth variants; write up losers before deleting them; rank by beam-1
- [Growth policies stay](growth-policies-stay.md) — adopting a split/link policy winner; change the default, never delete the loser
