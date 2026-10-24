---
name: torch-pinned-environment
description: Adding any dependency can silently upgrade numpy to 2.x and break the pinned torch 2.2.2; tests run on CPU and miss MPS-only bugs
metadata:
  type: convention
---

**Why:** torch 2.2.2 (the last Intel-macOS wheel) is built against numpy 1.x. A new dependency that pulls numpy 2.x breaks every `import torch` at runtime, even though the install itself succeeds.
**How to apply:** When you add or upgrade a dependency in pyproject.toml, or touch device-specific tensor code.

- After any `pip install`, confirm `numpy<2` still holds. Cap the new package to a version that supports numpy 1.x (for example, matplotlib is held at `<3.10`).
- The test suite forces `--device cpu`. MPS rejects some ops that CPU accepts, such as float64 (`.double()`), so code that builds analysis or report tensors must move them to the CPU first. Smoke-test new code on the default device once, not only in tests.
