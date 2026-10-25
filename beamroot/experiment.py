"""Run a grid of DAG training runs (datasets × seeds) and summarise them.

    python -m beamroot.experiment default [--datasets mnist] [--seeds 0 1]
        [--name NAME] [--device cpu]

Every run uses `--model dag --merge`. Runs go to runs/experiments/<name>/, one
directory per dataset and seed; a run with model.pt and eval.json is finished and
skipped on a rerun. Writes summary.json and summary.md (mean ± std per dataset).

Experiment 0001's loss × split-init grid ("routing") is gone with those modes
(ADR 0003); run_metrics still reads its run directories.
"""
import argparse
import json
import time
from itertools import product
from pathlib import Path
from statistics import mean, stdev

from beamroot import train

# A grid may add "variants": {name: [train flags]}; the summary groups by dataset × variant.
# Grids without variants run each dataset × seed once and show no variant column.
# "epochs": {dataset: n} overrides the dataset's default epochs (--epochs overrides both).
_SPLIT_FLAGS = ["--max-nodes", "19"]
GRIDS = {
    "default": {"datasets": ["mnist", "cifar10"], "seeds": [0, 1, 2]},
    # Experiment 0002: split policies.
    "splits": {"datasets": ["mnist", "cifar10"], "seeds": [0, 1, 2],
               "epochs": {"mnist": 12, "cifar10": 15},
               "variants": {
                   "binary": [*_SPLIT_FLAGS, "--max-k", "2"],
                   "kway-cluster": [*_SPLIT_FLAGS, "--max-k", "4", "--split-policy", "cluster"],
                   "kway-class": [*_SPLIT_FLAGS, "--max-k", "4", "--split-policy", "class"]}},
}
METRICS = ["beam1_acc", "beam3_acc", "soft_acc", "gap", "leaves", "mean_k", "links",
           "beam1_nodes", "split_router_entropy", "leaf_purity", "merges",
           "closest_merge_kl", "train_minutes"]
HEADERS = ["beam-1 acc", "beam-3 acc", "soft acc", "soft − beam-1", "leaves", "mean k",
           "links", "beam-1 nodes", "split-router entropy", "leaf purity",
           "merges", "closest merge KL", "train min"]


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean_or_none(values):
    values = list(values)
    return mean(values) if values else None


def _count(links) -> int:
    return links if isinstance(links, int) else len(links)


def run_metrics(run_dir) -> dict:
    """Table metrics of one finished run, read from its files only."""
    run_dir = Path(run_dir)
    evaluated = json.loads((run_dir / "eval.json").read_text())
    rows = {r["mode"]: r["accuracy"] for r in evaluated}
    nodes = {r["mode"]: r.get("mean_nodes") for r in evaluated}
    extra = json.loads((run_dir / "experiment.json").read_text())
    merges, kls, ks, links = 0, [], [], 0
    for event in _jsonl(run_dir / "growth.jsonl"):
        if "split" in event:  # k and links are absent from binary, link-free runs
            ks.append(event.get("k", 2))
            links += _count(event.get("links", 0))
        if "merge" not in event:
            continue
        if event["merge"] is None:
            kls.extend(event.get("candidates", {}).values())
        else:
            merges += 1
            kls.append(event["kl"])
    beam1, soft = rows["beam 1"], rows["soft"]
    return {
        "beam1_acc": beam1,
        "beam3_acc": rows.get("beam 3"),
        "soft_acc": soft,
        "gap": soft - beam1,
        "leaves": extra.get("leaves", len(extra["leaf_purity"])),
        "mean_k": _mean_or_none(ks),
        "links": links,
        "beam1_nodes": nodes.get("beam 1"),
        "split_router_entropy": _mean_or_none(
            e for r, e in extra["router_entropy"].items() if r != "root"),
        "leaf_purity": _mean_or_none(extra["leaf_purity"].values()),
        "merges": merges,
        "closest_merge_kl": min(kls) if kls else None,
        "train_minutes": extra["train_minutes"],
    }


def run_name(dataset: str, seed: int, variant: str | None = None) -> str:
    return f"{dataset}-seed{seed}" + (f"-{variant}" if variant else "")


def finished_run(exp_dir: Path, name: str) -> Path | None:
    """Latest run directory for `name` holding model.pt and eval.json, else None."""
    done = [p for p in sorted(exp_dir.glob(f"*-{name}"))
            if (p / "model.pt").exists() and (p / "eval.json").exists()]
    return done[-1] if done else None


def measure(run_dir: Path, device, limit_batches=None, train_minutes: float = 0.0) -> None:
    """Test-set reporting numbers for a trained run: experiment.json, then eval.json last
    (eval.json marks the run finished)."""
    from beamroot.analysis import class_flow, collect_stats, purity
    from beamroot.data import dataset_info, get_loaders
    from beamroot.evaluate import evaluate_modes, load_run
    model, config = load_run(run_dir, device)
    _, _, test_loader = get_loaders(config["dataset"], config["batch_size"], config["seed"],
                                     config.get("val_size", 0))
    rows = evaluate_modes(model, test_loader, device, (1, 3), limit_batches)
    stats = collect_stats(model, test_loader, device, model.num_classes, limit_batches)
    leaves = [n for n in model.graph.topological_order() if model.graph.nodes[n].kind == "leaf"]
    (run_dir / "flow.txt").write_text(class_flow(
        model.graph, stats["class_mass"], dataset_info(config["dataset"]).get("classes")) + "\n")
    (run_dir / "experiment.json").write_text(json.dumps({
        "train_minutes": train_minutes,
        "leaves": len(leaves),
        "router_entropy": stats["entropy"],
        "leaf_purity": {n: purity(stats["class_mass"][n])[1] for n in leaves}}, indent=2))
    (run_dir / "eval.json").write_text(json.dumps(rows, indent=2))


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None}
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def summarise(runs: list[dict], datasets: list[str]) -> list[dict]:
    """Mean ± std per dataset × variant: datasets in `datasets` order, best beam-1 first
    within a dataset."""
    groups: dict = {}
    for r in runs:
        groups.setdefault((r["dataset"], r.get("variant")), []).append(r)
    rows = [{"dataset": d, "variant": v, "n": len(rs),
             **{m: _stats(r[m] for r in rs) for m in METRICS}}
            for (d, v), rs in groups.items()]
    return sorted(rows, key=lambda r: (datasets.index(r["dataset"]),
                                       -(r["beam1_acc"]["mean"] or 0)))


def _cell(s: dict, digits: int) -> str:
    if s["mean"] is None:
        return "–"
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def format_table(rows: list[dict]) -> str:
    digits = {"merges": 1, "train_minutes": 1, "leaves": 1, "mean_k": 2, "links": 1,
              "beam1_nodes": 2}
    with_variant = any(r.get("variant") for r in rows)
    head = ["dataset"] + (["variant"] if with_variant else []) + ["n"] + HEADERS
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["dataset"]] + ([r["variant"] or "–"] if with_variant else []) + [str(r["n"])]
        cells += [_cell(r[m], digits.get(m, 4)) for m in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("grid", choices=sorted(GRIDS))
    p.add_argument("--datasets", nargs="+", default=None,
                   choices=["mnist", "cifar10", "synthetic", "synthetic_cifar"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--name", default=None, help="experiment directory name (default: the grid)")
    p.add_argument("--runs-dir", default="runs/experiments")
    p.add_argument("--device", default=None, help="mps or cpu (default: mps if available)")
    p.add_argument("--epochs", type=int, default=None, help="default depends on the dataset")
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--report", action="store_true",
                   help="also render each run's report.html (adds ~3 min per real run)")
    args = p.parse_args(argv)
    grid = GRIDS[args.grid]
    for axis in ("datasets", "seeds"):
        if getattr(args, axis) is None:
            setattr(args, axis, grid[axis])
    args.variants = grid.get("variants", {None: []})
    args.grid_epochs = grid.get("epochs", {})
    return args


def main(argv=None) -> Path:
    args = parse_args(argv)
    from beamroot.device import select_device
    device = select_device(args.device)
    exp_dir = Path(args.runs_dir) / (args.name or args.grid)
    exp_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for dataset, variant, seed in product(args.datasets, args.variants, args.seeds):
        name = run_name(dataset, seed, variant)
        run_dir = finished_run(exp_dir, name)
        if run_dir is None:
            print(f"== {name}", flush=True)
            argv = ["--model", "dag", "--merge", "--dataset", dataset, "--seed", str(seed),
                    "--name", name,
                    "--runs-dir", str(exp_dir)]
            argv += args.variants[variant]
            argv += [] if args.report else ["--no-report"]
            epochs = args.epochs if args.epochs is not None else args.grid_epochs.get(dataset)
            for flag, value in [("--device", args.device), ("--epochs", epochs),
                                ("--limit-batches", args.limit_batches)]:
                argv += [flag, str(value)] if value is not None else []
            t0 = time.time()
            run_dir = Path(train.run(train.parse_args(argv))["run_dir"])
            measure(run_dir, device, args.limit_batches, (time.time() - t0) / 60)
        else:
            print(f"== {name}: finished, skipping ({run_dir.name})", flush=True)
        runs.append({"dataset": dataset, "variant": variant, "seed": seed, "run_dir": str(run_dir),
                     **run_metrics(run_dir)})
    rows = summarise(runs, args.datasets)
    (exp_dir / "summary.json").write_text(json.dumps(
        {"grid": args.grid, "runs": runs, "rows": rows}, indent=2))
    table = format_table(rows)
    (exp_dir / "summary.md").write_text(table)
    print(table, flush=True)
    return exp_dir


if __name__ == "__main__":
    main()
