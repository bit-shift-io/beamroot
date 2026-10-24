# Progress — dag-router-prototype

## Project mechanics
Test: .venv/bin/python -m pytest -q   (venv created in slice 01; see README for interpreter version)
Type-check: n/a (Python, no type-checker configured)
Lint: none configured

## Slice 01 — scaffold-and-baseline
- Built: `beamroot` package, baseline CNN (stem + MLP head), MNIST training CLI, per-run dirs with config.json / metrics.jsonl / model.pt. Real default MNIST run: 99.20% test acc, 3:44 wall incl. download (MPS).
- Files touched: pyproject.toml, README.md, .gitignore, beamroot/{__init__,data,device,runs,train}.py, beamroot/models/{__init__,baseline,stem}.py, tests/test_baseline.py, tests/test_runs.py
- New exports/interfaces:
  - `beamroot.models.stem.build_stem(in_channels=1, width=32) -> nn.Sequential` — reusable Stem, two conv+pool blocks, 28×28 → 7×7, `2*width` channels out
  - `BaselineCNN(in_channels=1, image_size=28, num_classes=10, width=32)` with `.stem`, `.head`
  - `beamroot.device.select_device(requested: str | None) -> torch.device`
  - `beamroot.data.get_loaders(dataset, batch_size, seed) -> (train_loader, test_loader)`; datasets `mnist`, `synthetic` (offline, for tests)
  - `beamroot.runs.create_run_dir(name, root="runs", config=None) -> Path`, `log_metrics(run_dir, record)`
  - `beamroot.train.parse_args(argv)`, `run(args) -> dict` (keys run_dir, first_loss, loss, test_acc, epoch, seconds), plus `train_epoch`, `evaluate`
  - CLI flags: --dataset --model --device --seed --epochs (default 5) --batch-size --lr --limit-batches --name --runs-dir
- Conventions confirmed: tests in tests/test_*.py use `synthetic` dataset, `--device cpu`, `--runs-dir tmp_path`, and call `run(parse_args([...]))` directly. Install in zsh: `pip install -e '.[dev]'`.
- Surprises: machine is Intel x86_64 (i7-9750H), not Apple silicon, though MPS reports available. Last macOS x86_64 torch wheels are 2.2.2 for Python ≤3.12, so venv uses /usr/local/bin/python3.12, torch==2.2.2, torchvision==0.17.2, numpy<2; requires-python `>=3.12,<3.13`. Baseline used 3:44 of the 5-min MNIST budget — little headroom for the DAG model.
- Built by: slice-implementer · sonnet · 19226 · verification passed first time

## Slice 02 — router-graph-soft-routing
- Built: DAG router graph (structure, add/remove, cycle check, topological order, serialisation), RouterNode/Leaf heads on Stem features, soft log-space forward with per-node outputs, `--model dag` in train CLI with graph-carrying checkpoints. Real MNIST fixed 3-node graph: 99.12% in 3:51 (MPS). Run: runs/20261024-142927-mnist-dag-slice02/
- Files touched: beamroot/models/{graph,nodes,dag}.py, beamroot/train.py, tests/test_graph.py, tests/test_dag_forward.py
- New exports/interfaces:
  - `beamroot.models.graph.RouterGraph()`: `.nodes: dict[str, Node(name, kind, parents: list[str])]`, `.edges: list[tuple[str,str]]`; `add_node(name, kind "router"|"leaf", parents=None)`, `add_edge(parent, child)` (ValueError on cycle / leaf parent / duplicate / self-loop), `remove_node(name)`, `parents(n)`, `children(n)` (edge-insertion order = router output index), `descendants(n)`, `root`, `routers()`, `leaves()` (topological), `topological_order()` (Kahn, insertion-order ties), `to_dict()` / `RouterGraph.from_dict(d)`. Node names must not contain ".".
  - `beamroot.models.nodes.RouterNode(in_features, num_children, hidden=64)` → log-softmax (B,k), final layer `.out`; `Leaf(in_features, num_classes, hidden=128)` → logits.
  - `beamroot.models.dag.default_graph()` (root→leaf0, leaf1); `DAGRouterModel(graph=None, in_channels=1, image_size=28, num_classes=10, width=32)` with `.stem`, `.graph`, `.nodes: nn.ModuleDict`, `.config`, `.in_features`; `build_node(name)`; `route(x)` → `{log_reach {node:(B,)}, log_route {router:(B,k)}, leaf_log_probs {leaf:(B,C)}, log_probs (B,C)}`; `forward(x)` → log_probs; `checkpoint()` → `{config, graph, state_dict}`; `DAGRouterModel.from_checkpoint(ckpt)`.
  - `beamroot.train.build_model(name)`, `save_payload(model)`; `--model baseline|dag`.
- Conventions confirmed: no tests/__init__.py — test modules can't import each other; shared fixtures are copied per file. forward returns log-probs (cross_entropy/argmax still work).
- Surprises: graph ops do not update `model.nodes` — callers must add/remove/rebuild modules via `build_node`; a router's output size is fixed at build. Checkpoints saved on MPS need `torch.load(..., map_location='cpu')`. 3-node DAG already uses 3:51 of the 5-min MNIST budget.
- Built by: slice-implementer · opus (inherited) · 28982 · verification passed first time

## Slice 03 — beam-search-inference
- Built: DAG beam search (paths reaching the same node add up; each node runs at most once; a router expands only once no other beam node can still reach it), plus `predict` (top-N paths per image) and `evaluate` (mode / accuracy / mean nodes table) CLIs. Real slice-02 run: soft 0.9912 (3 nodes), beam1 0.9910 (2 nodes), beam3/5 = soft.
- Files touched: beamroot/{beam,predict,evaluate}.py, tests/test_beam.py
- New exports/interfaces:
  - `beamroot.beam.beam_search(model, x, beam: int) -> list[BeamResult]` (one per image)
  - `BeamResult(paths: list[BeamPath], probs: Tensor(C,), nodes_evaluated: int)` with `.pred`; class probs = kept leaves mixed by renormalised probability
  - `BeamPath(leaf, nodes: list[str], prob, path_prob, pred)` — `nodes` = most probable incoming path, `prob` = combined prob, `path_prob` = that path's prob
  - `beamroot.evaluate.load_run(run_dir, device) -> (model, config)`, `evaluate_modes(model, loader, device, beams=(1,3,5), limit_batches=None) -> [{mode, accuracy, mean_nodes}]`, `format_table(rows)`, `main(argv) -> rows`; flags --beams --device --limit-batches. Writes `<run-dir>/eval.json`.
  - `beamroot.predict.format_result(index, label, res)`, `main(argv) -> [(label, BeamResult)]`; flags --beam --samples --device (first N unshuffled test images)
- Conventions confirmed: node calls counted with forward hooks, not mocks; CLI tests build tiny synthetic dag runs in tmp_path.
- Surprises: in unusual DAGs the beam can hold more nodes than there are leaves, so "beam ≥ leaf count == soft" is only guaranteed on trees and the tested DAGs. `evaluate` writes eval.json (not in plan).
- Built by: slice-implementer · opus (inherited) · 38093 · verification passed first time

## Slice 04 — load-balancing
- Built: per-router load-balancing loss `reach * KL(usage || uniform)` summed over routers, scaled by `--balance-weight` (CLI default 0.1, 0 disables); per-epoch `routing` stats in metrics.jsonl. Real MNIST: root usage [0.506, 0.494], 0.9911 acc, ~3:36. Run: runs/20261024-143843-mnist-dag-slice04/
- Files touched: beamroot/losses.py (new), beamroot/train.py, tests/test_losses.py (new); dag.py unchanged (route() already exposes log_reach/log_route)
- New exports/interfaces:
  - `beamroot.losses.balance_loss(log_reach: dict, log_route: dict) -> Tensor`, `router_usage(log_reach (B,), log_route (B,k)) -> (k,)`, `routing_entropy(log_route) -> scalar`
  - `train_epoch(model, loader, opt, device, limit_batches=None, balance_weight=0.0, stats=None)` — `stats` filled with `{router: {"usage": [..], "entropy": float}}`
  - metrics record gains `routing: {router: {usage, entropy}}` for dag only
- Conventions confirmed: baseline records have no `routing` key and no balance loss.
- Surprises: logged dag `loss` = CE + weighted balance term. `train_epoch` default balance_weight 0.0 vs CLI 0.1. Usage/entropy averaged over training steps, not test set. Weight-0 parity with slice 02 checked by test + code reading, not a full MNIST run. Unreachable-router weighting only checked on a synthetic router.
- Built by: slice-implementer · sonnet (low) · 22115 · verification passed first time

## Slice 05 — growth-by-splitting
- Built: per-leaf reach-weighted loss, split op (graph + model), Adam rebuild keeping surviving params' state, growth schedule in train with growth.jsonl + graph snapshots. Default dag MNIST run grows 3→5→7 nodes; beam1 0.9883, beam3/5/soft 0.9898; train wall 4:12. Run: runs/20261024-144555-mnist-dag-slice05/
- Files touched: beamroot/growth.py (new), beamroot/models/graph.py, beamroot/train.py, tests/test_growth.py (new), tests/test_dag_forward.py (added --grow-every 0 to the fixed-graph CLI test)
- New exports/interfaces:
  - `RouterGraph.split(name) -> (new_a, new_b)` — ValueError if not a leaf; node becomes router, keeps its name and all parents; new leaves get lowest free `leaf<i>`
  - `beamroot.growth.leaf_losses(model, loader, device, limit_batches=None) -> {leaf: float}` (reach × leaf NLL, eval mode), `choose_leaf(losses) -> str`, `split_leaf(model, name, noise=1e-3) -> (a, b)`, `rebuild_optimizer(model, opt) -> opt`, `save_graph_snapshot(model, run_dir, epoch) -> Path` (`<run>/graphs/epoch_NNN.json`, epoch_000 = start), `grow(model, loader, device, epoch, run_dir, max_nodes, limit_batches=None, noise=1e-3) -> event | None`
  - CLI: `--grow-every` (default 1, 0 disables; dag only, never after last epoch), `--max-nodes` (default 7, counts all nodes; +2 per split), `--growth-batches` (default 50)
  - growth.jsonl lines: epoch, split node, new node ids, num_nodes, per-leaf losses. metrics.jsonl gains `num_nodes` for dag.
- Conventions confirmed: `diamond()` helper copied per test file.
- Surprises: (1) growth on by default broke the fixed-graph CLI test → it now passes `--grow-every 0`. (2) Split-created routers don't specialise: ~ln2 entropy, ~50/50 usage on MNIST — new children start near-identical and the balance loss favours uniform. Beam1 0.9883 vs soft 0.9898. Larger split noise or lower balance weight for new routers might fix it — open for human decision. (3) README.md doesn't document the new flags.
- Built by: slice-implementer · opus (inherited) · 44649 · verification passed first time

## Slice 06 — run-report
- Built: reach-weighted per-node class mix, purity (optional class→group mapping), mean edge routing probs, longest-path depth layout; graph.png + self-contained report.html (base64); auto-report at end of dag train runs (`--no-report` skips; baseline runs skip). Real slice-05 run report: 28s on CPU (mostly evaluate_modes on 10k images). 31-node tree renders legibly (2004x597).
- Files touched: beamroot/analysis.py, beamroot/report.py, beamroot/templates/report.html, tests/test_analysis.py, tests/test_report.py, beamroot/train.py (hook + flag), pyproject.toml (deps + package data)
- New exports/interfaces:
  - `beamroot.analysis.node_depths(graph) -> {node: depth}`, `purity(class_mass, groups=None) -> (dominant class|group, share)` (empty → (None, 0.0)), `collect_stats(model, loader, device, num_classes, limit_batches=None) -> {"class_mass": {node: [C]}, "edge_prob": {(parent, child): float}}`
  - `beamroot.report.render_report(run_dir, device=None, samples=8, beam=3, beams=(1,3,5), limit_batches=None, groups=None, purity_label="class") -> {"graph": Path, "report": Path}`; `layout(graph)`, `node_labels(graph, class_mass, groups=None)`, `draw_graph(graph, edge_prob, labels, path=None) -> png bytes`, `draw_histograms(graph, class_mass)`
  - CLI `python -m beamroot.report <run-dir> [--device --samples --beam --limit-batches]`; train `--no-report`
- Conventions confirmed: matplotlib Agg backend, no Graphviz binary. pyproject pins `numpy<2`, `matplotlib<3.10` (3.9.4), plus networkx, jinja2; templates/*.html declared as package data.
- Surprises: `pip install matplotlib` pulled numpy 2.x, which breaks torch 2.2.2 — any new dep must keep numpy<2. Auto-report adds ~28s to a dag MNIST run (eats into 5-min budget). Pytest now shows 14 PyparsingDeprecationWarnings from matplotlib internals (harmless). Diagram makes the slice-05 50/50 split-router problem visible.
- Built by: slice-implementer · sonnet · 47490 · verification passed first time
