# beamroot

## Setup

Use **Python 3.12**. On Intel macOS the last PyTorch wheels are 2.2.2 and only
exist for Python <= 3.12 (no 3.13/3.14 wheels), so do not use the system 3.14.

```
python3.12 -m venv .venv
.venv/bin/pip install "torch==2.2.2" "torchvision==0.17.2" "numpy<2"   # Intel Mac pins
.venv/bin/pip install -e .[dev]
.venv/bin/python -m pytest -q
```

On Apple silicon, plain `pip install -e .[dev]` with Python 3.12 is enough.

## Train

```
.venv/bin/python -m beamroot.train --dataset mnist --model baseline
```

Options: `--device {mps,cpu}` (default: MPS if available, else CPU), `--seed`,
`--epochs`, `--batch-size`, `--lr`, `--limit-batches N` (smoke runs), `--name`.
Each run writes `runs/<timestamp>-<name>/` with `config.json`, `metrics.jsonl`, `model.pt`.
MNIST downloads to `data/` on first use.
