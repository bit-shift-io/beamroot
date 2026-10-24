# DAG Router Prototype

## Problem
- Standard image classifiers run every neuron on every image and return one label with no explanation.
- Interpretability research shows networks contain specialised features, but the architecture does not expose them.
- We want to test whether a network built as a graph of small routing decisions can learn, grow itself, and explain its answers.

## Solution
- A **router graph**: a shared conv stem feeds a DAG of small **router nodes**, each giving probabilities over its children; **leaves** classify.
- The graph **grows** during training by splitting the worst leaf; a later slice adds **merging** of near-identical nodes.
- **Beam search** at inference returns the top-N **paths**, each with its probability and leaf prediction, plus an HTML report of what each node learned.

## Out of Scope
- Datasets other than MNIST and CIFAR-10.
- Jupyter notebooks, web UI, or a serving API.
- Hyperparameter search, distributed training, or CUDA-specific tuning.
- Learned growth policies (growth rules are hand-written heuristics).

## Cross-cutting Constraints
- Python package `beamroot` with CLI scripts; no notebooks.
- Device selection: MPS if available, else CPU. Every script accepts `--device`.
- Every run writes to its own `runs/<timestamp>-<name>/` directory: config, metrics, checkpoint, artefacts.
- Runs are reproducible from `--seed`.
- Default MNIST run finishes in under ~5 minutes on an Apple-silicon Mac; CIFAR-10 in under ~20 minutes.
- The graph is a DAG in code at all times: nodes hold a list of parents, never a single parent (see ADR 0001).

## Acceptance Criteria
- [ ] `python -m beamroot.train --dataset mnist --model dag` grows a graph and reaches ≥97% test accuracy at beam 1.
- [ ] `--model baseline` trains a plain CNN with a parameter count within 10% of the final DAG model.
- [ ] Beam search prints top-N paths with probabilities; accuracy and nodes-evaluated are reported for N = 1, 3, 5.
- [ ] Each run produces a graph diagram and an HTML report showing sample paths and per-node class mix.
- [ ] CIFAR-10 runs end to end; the report shows per-node purity for animal vs vehicle groups.
- [ ] `--merge` enables node merging; a run with merging completes and its report shows merged nodes with multiple parents.

## References
- [ADR 0001: Router graph is a DAG](../../docs/adr/0001-router-graph-is-a-dag.md)
- [Glossary](../../CONTEXT.md)
- Prior art: Frosst & Hinton 2017 (soft decision trees); Tanno et al. 2019 (Adaptive Neural Trees); Mixture-of-Experts load balancing (Shazeer et al. 2017).
