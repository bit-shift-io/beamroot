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
.venv/bin/python -m beamroot.train --dataset mnist
```

This trains the growing router DAG (`--model dag`, the default). Add `--model baseline` for the plain CNN to compare against.

Each run writes `runs/<timestamp>-<name>/` with `config.json`, `metrics.jsonl`, `model.pt`.
MNIST downloads to `data/` on first use. `--name` defaults to `<dataset>-<model>`.

Flags (defaults in brackets; the dataset defaults are in `beamroot/data.py`):

- `--dataset {mnist,synthetic,cifar10,synthetic_cifar}` [mnist]
- `--model {dag,baseline}` [dag]
- `--device DEVICE` [mps if available, else cpu]
- `--seed SEED` [0]. Also chooses the validation images.
- `--epochs EPOCHS` [mnist 5, cifar10 15, synthetic 5, synthetic_cifar 5]
- `--width WIDTH` stem width [32 for every dataset]
- `--hidden HIDDEN` baseline head hidden size [128]
- `--match-params RUN_DIR` baseline only: size the CNN within 10% of the DAG run's parameter count
- `--val-size VAL_SIZE` training images held out as the validation set for merges [5000]. Use 0 to disable; `--merge` then refuses to run.
- `--batch-size BATCH_SIZE` [128]
- `--lr LR` [0.001]
- `--balance-weight BALANCE_WEIGHT` load-balancing loss weight, dag only (0 disables) [0.1]
- `--grow-every GROW_EVERY` dag only: split the worst leaf every K epochs (0 disables) [1]
- `--max-nodes MAX_NODES` dag only: stop growing when a split would exceed this many graph nodes [7]
- `--preset {explainable,pure}` dag only: growth defaults for `--max-k`, `--split-policy` and `--link-policy` [explainable]. `explainable` (unsupervised) = `--max-k 4 --split-policy cluster`; `pure` (supervised, uses labels) = `--max-k 4 --split-policy class`. An explicit `--max-k` or `--split-policy` overrides the preset; `--max-k 2` gives binary splits. Chosen in [experiment 0002](docs/experiments/0002-kway-splits.md), [ADR 0004](docs/adr/0004-growth-presets.md)
- `--max-k MAX_K` dag only: a split may make up to this many new leaves. It clusters the leaf's activations for k = 2…MAX_K and keeps 2 unless a larger k has a clearly higher silhouette (by more than 0.05). k is capped by what fits under `--max-nodes`; `growth.jsonl` records `k` and `silhouettes` [from `--preset`: 4]
- `--split-policy {cluster,class}` dag only [from `--preset`: `cluster`]: how a split groups the leaf's images. `class` gives each class with at least 10% of the leaf's reach its own group and the rest one shared group (at most `--max-k` groups; the smallest qualifying classes fold into the shared one). If no class qualifies it falls back to `cluster`. `growth.jsonl` records `policy`, `groups` (classes per group) and `policy_fallback`.
- `--link-policy {none,loss,class}` dag only [from `--preset`: `none`]: whether a split group may link to an existing leaf instead of getting a fresh one. `loss`: before the split, the leaf's images in the validation set join the groups (nearest cluster centre, or by label under `class`); a group links to the existing leaf with the lowest mean NLL on them if that is below 0.5 × the split leaf's. At least one group stays fresh; groups linking to one leaf share one router output. `class`: a group links to an existing leaf whose validation dominant class equals the group's and whose class purity is ≥ 0.9 (highest purity wins; the lowest-purity link stays fresh if all would link); `links` then holds `{leaf, class, purity}`. Needs `--val-size > 0`. `growth.jsonl` records `links` {group: {leaf, ratio}} and `link_ratios` (each group's best candidate); `new_nodes` lists only the fresh leaves.
- `--growth-batches GROWTH_BATCHES` training batches used to measure per-leaf loss and to cluster the chosen leaf's images before a split [50]
- `--split-noise SPLIT_NOISE` std of the noise added to the kept units of a split's new leaves [0.001]
- `--merge` dag only: at each growth step, first merge node pairs whose outputs on the validation set are near-identical [off]
- `--merge-threshold MERGE_THRESHOLD` merge pairs whose mean symmetric KL is below this, in nats [0.05]
- `--no-report` dag only: skip writing `graph.png` and `report.html` at the end of the run [off, so the report is written]
- `--limit-batches LIMIT_BATCHES` cap batches per epoch, for smoke runs [no cap]
- `--name NAME` run name [`<dataset>-<model>`]
- `--runs-dir RUNS_DIR` [runs]

Examples:

Growing DAG run with cross-depth merges (about 7 min on MNIST on Apple silicon, including the report):

```
.venv/bin/python -m beamroot.train --dataset mnist --merge --name mnist-dag
```

CIFAR-10 DAG run, a baseline CNN sized to match it, and a report with the baseline's accuracy beside the DAG's (about 16 min for the DAG on Apple silicon):

```
.venv/bin/python -m beamroot.train --dataset cifar10 --merge --name cifar-dag
.venv/bin/python -m beamroot.train --dataset cifar10 --model baseline --match-params runs/<timestamp>-cifar-dag --name cifar-baseline
.venv/bin/python -m beamroot.report runs/<timestamp>-cifar-dag --baseline runs/<timestamp>-cifar-baseline
```

DAG models train on the path loss and grow by cluster splits (see [ADR 0003](docs/adr/0003-path-loss-and-cluster-splits.md)). By default a split may make up to 4 children, chosen by silhouette (the `explainable` preset, [ADR 0004](docs/adr/0004-growth-presets.md)).

## Evaluate

```
.venv/bin/python -m beamroot.evaluate runs/<timestamp>-mnist-dag
```

Reports test accuracy in two modes and writes `eval.json`:

- **Beam N** keeps the N most probable partial paths and answers from the best one. **Beam 1** is one path and one leaf, one answer. It is the main measure of a model.
- **Soft** runs every node and mixes all leaves by reach. It is reported for comparison, not as the answer. The gap between soft and beam 1 shows how much the answer leans on mixing.

Flags:

- `run_dir` positional: the run directory to evaluate
- `--beams BEAMS [BEAMS ...]` beam widths to evaluate [1 3 5]
- `--device DEVICE` [mps if available, else cpu]
- `--limit-batches LIMIT_BATCHES` cap test batches [no cap]

## Predict

```
.venv/bin/python -m beamroot.predict runs/<timestamp>-mnist-dag --beam 1 --samples 3
```

Prints, for the first test images, the beam prediction and each kept path with its probability and class.

Flags:

- `run_dir` positional: the run directory to load
- `--beam BEAM` beam width [3]
- `--samples SAMPLES` number of test images [5]
- `--device DEVICE` [mps if available, else cpu]

## Report

```
.venv/bin/python -m beamroot.report runs/<timestamp>-mnist-dag
```

Writes `graph.png` and `report.html` (images embedded) into the run directory. A DAG run writes them automatically at the end of training unless `--no-report` is set.

Flags:

- `run_dir` positional: the run directory to report on
- `--device DEVICE` [mps if available, else cpu]
- `--samples SAMPLES` images shown [8]
- `--beam BEAM` beam width used for the report [3]
- `--limit-batches LIMIT_BATCHES` cap test batches used for the accuracy and statistics [no cap]
- `--baseline RUN_DIR` baseline run directory to show beside the DAG accuracy [none]

## Experiment

```
.venv/bin/python -m beamroot.experiment default
```

Runs the `default` grid: every dataset in the grid × every seed, each as `train --model dag --merge --no-report`. Runs go to `runs/experiments/<name>/`, one directory per dataset and seed. A run that already has `model.pt` and `eval.json` is skipped, so a rerun resumes. The default grid is 6 runs: about 4 min of training each on MNIST and about 13 min on CIFAR-10 on Apple silicon, so roughly 50 min in total without `--report`. It ends by writing `summary.json` and `summary.md` (mean ± std per dataset, sorted by beam-1 accuracy within a dataset). Columns: beam-1, beam-3 and soft accuracy, soft − beam-1, leaf count, mean k per split, links, mean nodes evaluated by beam 1, plus the routing diagnostics. Every run also writes `flow.txt` (the class flow tree).

A grid entry in `GRIDS` may add `"variants": {name: [train flags]}` (e.g. `["--max-nodes", "19"]`). Each variant runs for every dataset × seed, run directories get the variant as a suffix, and the summary groups by dataset × variant with an extra `variant` column. Grids without variants (like `default`) are unchanged.

Flags:

- `grid {default}` positional: the grid to run. `default` is datasets mnist and cifar10 × seeds 0, 1, 2.
- `--datasets {mnist,cifar10,synthetic,synthetic_cifar} [...]` narrow the grid to these datasets
- `--seeds SEEDS [SEEDS ...]` narrow the grid to these seeds
- `--name NAME` experiment directory name [the grid name, `default`]
- `--runs-dir RUNS_DIR` [runs/experiments]
- `--device DEVICE` [mps if available, else cpu]
- `--epochs EPOCHS` [per dataset, as for `train`]
- `--limit-batches LIMIT_BATCHES` cap batches per epoch, for smoke runs [no cap]
- `--report` also render each run's `report.html` (adds about 3 min per real MNIST run) [off]

The results and their interpretation are in the experiment log: [docs/experiments/README.md](docs/experiments/README.md) and [0001: Routing specialisation](docs/experiments/0001-routing-specialisation.md).
