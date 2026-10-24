Status: pending
Complexity: 4 (design 1, scope 2, tests 1, hazards 0)
MCP: none

# Scaffold and baseline CNN on MNIST

## What to build
- Installable `beamroot` package with venv setup documented in the repo README.
- `python -m beamroot.train --dataset mnist --model baseline` trains a plain CNN and prints per-epoch loss and test accuracy.
- Each run writes `runs/<timestamp>-<name>/` with `config.json`, `metrics.jsonl` and `model.pt`.

## Files to create/modify
- pyproject.toml
- README.md
- .gitignore (`.venv/`, `data/`, `runs/`)
- beamroot/__init__.py
- beamroot/data.py (MNIST loaders, auto-download to `data/`)
- beamroot/device.py (MPS → CPU selection)
- beamroot/models/baseline.py
- beamroot/train.py (CLI, training loop, run directory)
- beamroot/runs.py (run dir creation, metrics logging)
- tests/test_baseline.py
- tests/test_runs.py

## Test approach
- Baseline forward pass returns `[batch, 10]` logits on random input.
- Training loop on a tiny synthetic dataset reduces loss over a few steps.
- Run directory contains config, metrics and checkpoint after a 1-epoch run with `--limit-batches`.
- Seed makes two tiny runs produce identical first-step loss.

## Acceptance criteria
- [ ] `pip install -e .[dev]` and `pytest` pass on a fresh venv.
- [ ] Baseline reaches ≥98% MNIST test accuracy in the default epochs.
- [ ] `--device cpu` and `--seed` work.
- [ ] `--limit-batches N` exists for fast smoke runs.

## Blocked by
None — can start immediately.

## Gotchas
- Python 3.14 is installed. If PyTorch has no 3.14 wheel, create the venv with 3.12/3.13 and note it in README.
- Tests must not download datasets; use synthetic tensors.
