---
name: torch-pinned-environment
description: Adding any dependency can silently upgrade numpy to 2.x and break the pinned torch 2.2.2; tests run on CPU and miss MPS-only bugs (no float64, no `index_copy`, adaptive pool needs divisible sizes)
metadata:
  type: convention
---

**Why:** torch 2.2.2 (the last Intel-macOS wheel) is built against numpy 1.x. A new dependency that pulls numpy 2.x breaks every `import torch` at runtime, even though the install itself succeeds.
**How to apply:** When you add or upgrade a dependency in pyproject.toml, or touch device-specific tensor code.

- After any `pip install`, confirm `numpy<2` still holds. Cap the new package to a version that supports numpy 1.x (for example, matplotlib is held at `<3.10`).
- The test suite forces `--device cpu`. MPS rejects some ops that CPU accepts, such as float64 (`.double()`), so code that builds analysis or report tensors must move them to the CPU first. Smoke-test new code on the default device once, not only in tests.
- Known MPS gaps hit by the cascade model: no float64 (`.double()` on a device tensor), no `Tensor.index_copy` (use clone plus `out[idx] = sub`), and `adaptive_avg_pool2d` requires the input size to be divisible by the output size (28 to 21 crashes; `cascade._avg_resize` is the exact replacement). None of these fail on CPU, and each one only showed up when a run used the default device.
- Run the MPS smoke yourself, with the report enabled, and confirm `torch.backends.mps.is_available()`. Subagents reported MPS smokes that never ran on MPS or skipped the report step, and two real bugs were hiding behind them.
