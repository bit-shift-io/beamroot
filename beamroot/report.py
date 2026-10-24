"""Graph diagram and self-contained HTML report for a trained DAG run.

    python -m beamroot.report <run-dir> [--device cpu] [--samples 8] [--beam 3]

Writes <run-dir>/graph.png and <run-dir>/report.html (images embedded as base64).
"""
import argparse
import base64
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import torch  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from beamroot.analysis import collect_stats, node_depths, purity  # noqa: E402
from beamroot.beam import beam_search  # noqa: E402
from beamroot.data import dataset_info, get_loaders  # noqa: E402
from beamroot.device import select_device  # noqa: E402
from beamroot.evaluate import evaluate_modes, load_run  # noqa: E402

TEMPLATE_DIR = Path(__file__).parent / "templates"


def layout(graph) -> dict[str, tuple[float, float]]:
    """Position nodes in rows by depth, spread evenly within a row (x in [0, 1])."""
    depths = node_depths(graph)
    rows: dict[int, list[str]] = {}
    for name in graph.topological_order():
        rows.setdefault(depths[name], []).append(name)
    pos = {}
    for d, names in rows.items():
        for i, name in enumerate(names):
            pos[name] = ((i + 0.5) / len(names), -float(d))
    return pos


def node_labels(graph, class_mass, groups=None) -> dict[str, str]:
    labels = {}
    for name in graph.nodes:
        dom, pur = purity(class_mass[name], groups)
        labels[name] = f"{name}\n{'-' if dom is None else dom} {pur:.0%}"
    return labels


def draw_graph(graph, edge_prob, labels, path=None) -> bytes:
    """Render the graph to PNG bytes (and to `path` if given). Size scales with the widest row."""
    pos = layout(graph)
    rows = {}
    for name, (_, y) in pos.items():
        rows[y] = rows.get(y, 0) + 1
    width_nodes = max(rows.values())
    fig_w = max(6.0, 1.6 * width_nodes)
    fig_h = max(3.0, 1.5 * len(rows))
    pos = {n: (x * fig_w, y * 1.5) for n, (x, y) in pos.items()}
    g = nx.DiGraph()
    g.add_nodes_from(graph.nodes)
    g.add_edges_from(graph.edges)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    colors = ["#cfe8ff" if graph.nodes[n].kind == "router" else "#d8f0d0" for n in g.nodes]
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowsize=10, edge_color="#666",
                           node_size=2600, node_shape="s")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_shape="s", node_size=2600,
                           edgecolors="#333")
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)
    nx.draw_networkx_edge_labels(
        g, pos, ax=ax, edge_labels={e: f"{p:.2f}" for e, p in edge_prob.items()},
        font_size=6, label_pos=0.35, rotate=False, bbox={"fc": "white", "ec": "none", "pad": 0.5})
    ax.margins(0.08)
    ax.axis("off")
    data = _png(fig)
    if path is not None:
        Path(path).write_bytes(data)
    return data


def draw_histograms(graph, class_mass) -> bytes:
    """Small multiples: one class histogram per node, in topological order."""
    names = graph.topological_order()
    cols = min(4, len(names))
    nrows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.0 * cols, 1.9 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        mass = class_mass[name]
        total = sum(mass)
        ax.axis("on")
        ax.bar(range(len(mass)), [m / total if total > 0 else 0 for m in mass], color="#4477aa")
        ax.set_title(name, fontsize=8)
        ax.set_xticks(range(len(mass)))
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    return _png(fig)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _image_png(img: torch.Tensor) -> bytes:
    """A (C,H,W) tensor as a min-max scaled PNG."""
    img = img.detach().cpu().float()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    arr = img.permute(1, 2, 0).squeeze(-1).numpy()
    buf = io.BytesIO()
    plt.imsave(buf, arr, format="png", cmap="gray" if arr.ndim == 2 else None)
    return buf.getvalue()


def baseline_summary(run_dir) -> dict:
    """{name, accuracy, params} of a finished baseline run (last logged test accuracy)."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    last = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            if line.strip()][-1]
    return {"name": run_dir.name, "accuracy": last["test_acc"], "params": config.get("params")}


def render_report(run_dir, device=None, samples=8, beam=3, beams=(1, 3, 5),
                  limit_batches=None, groups=None, purity_label="class", baseline=None) -> dict:
    """Write graph.png and report.html into `run_dir`; returns {"graph": Path, "report": Path}.

    `groups` (class index -> group label) switches purity from classes to class groups; by
    default the dataset's own groups are used (CIFAR-10: animal / vehicle).
    `baseline` is a baseline run directory whose final accuracy is shown beside the DAG.
    """
    run_dir = Path(run_dir)
    device = select_device(device) if not isinstance(device, torch.device) else device
    model, config = load_run(run_dir, device)
    _, test_loader = get_loaders(config["dataset"], config["batch_size"], config["seed"])
    graph = model.graph
    if groups is None:
        groups = dataset_info(config["dataset"])["groups"]
        if groups is not None:
            purity_label = "group"

    rows = evaluate_modes(model, test_loader, device, beams, limit_batches)
    stats = collect_stats(model, test_loader, device, model.num_classes, limit_batches)
    labels = node_labels(graph, stats["class_mass"], groups)
    purity_rows = [{"node": n, "kind": graph.nodes[n].kind,
                    "class": purity(stats["class_mass"][n])[0],
                    "class_purity": purity(stats["class_mass"][n])[1],
                    "group": purity(stats["class_mass"][n], groups)[0] if groups else None,
                    "group_purity": purity(stats["class_mass"][n], groups)[1]}
                   for n in graph.topological_order()]
    baseline_ctx = baseline_summary(baseline) if baseline else None
    graph_png = draw_graph(graph, stats["edge_prob"], labels, run_dir / "graph.png")
    hist_png = draw_histograms(graph, stats["class_mass"])

    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.extend(y.tolist())
        if len(ys) >= samples:
            break
    x = torch.cat(xs)[:samples]
    results = beam_search(model, x.to(device), beam)
    sample_ctx = [{"index": i, "label": ys[i], "pred": r.pred, "image": _b64(_image_png(x[i])),
                   "paths": r.paths} for i, r in enumerate(results)]

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape())
    html = env.get_template("report.html").render(
        run_name=run_dir.name, config=config, num_nodes=len(graph.nodes), rows=rows,
        graph_png=_b64(graph_png), hist_png=_b64(hist_png), samples=sample_ctx, beam=beam,
        purity_label=purity_label, purity_rows=purity_rows, has_groups=groups is not None,
        baseline=baseline_ctx, dag_params=config.get("params"))
    report = run_dir / "report.html"
    report.write_text(html)
    return {"graph": run_dir / "graph.png", "report": report}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--device", default=None, help="mps or cpu (default: mps if available)")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--baseline", default=None, metavar="RUN_DIR",
                   help="baseline run directory to show beside the DAG accuracy")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    out = render_report(args.run_dir, args.device, args.samples, args.beam,
                        limit_batches=args.limit_batches, baseline=args.baseline)
    for path in out.values():
        print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
