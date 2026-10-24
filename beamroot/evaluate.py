"""Compare beam search and soft routing on a trained DAG run.

    python -m beamroot.evaluate <run-dir> [--beams 1 3 5] [--device cpu]

Prints accuracy and mean nodes evaluated per mode, and writes them to
<run-dir>/eval.json.
"""
import argparse
import json
from itertools import islice
from pathlib import Path

import torch

from beamroot.beam import beam_search
from beamroot.data import get_loaders
from beamroot.device import select_device
from beamroot.models.dag import DAGRouterModel


def load_run(run_dir, device):
    """(model on device in eval mode, run config) from a DAG run directory."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    model = DAGRouterModel.from_checkpoint(ckpt).to(device).eval()
    return model, config


def evaluate_modes(model, loader, device, beams=(1, 3, 5), limit_batches=None) -> list[dict]:
    """One row per mode: {mode, accuracy, mean_nodes}. Soft routing evaluates every node."""
    correct = {b: 0 for b in beams}
    nodes = {b: 0 for b in beams}
    soft_correct = total = 0
    with torch.no_grad():
        for x, y in islice(loader, limit_batches):
            x = x.to(device)
            soft_correct += (model(x).argmax(1).cpu() == y).sum().item()
            for b in beams:
                for res, label in zip(beam_search(model, x, b), y.tolist()):
                    correct[b] += res.pred == label
                    nodes[b] += res.nodes_evaluated
            total += len(y)
    total = max(total, 1)
    rows = [{"mode": f"beam {b}", "accuracy": correct[b] / total, "mean_nodes": nodes[b] / total}
            for b in beams]
    rows.append({"mode": "soft", "accuracy": soft_correct / total,
                 "mean_nodes": float(len(model.graph.nodes))})
    return rows


def format_table(rows) -> str:
    lines = [f"{'mode':<8} {'accuracy':>9} {'mean nodes evaluated':>21}"]
    for r in rows:
        lines.append(f"{r['mode']:<8} {r['accuracy']:>9.4f} {r['mean_nodes']:>21.2f}")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--beams", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--device", default=None, help="mps or cpu (default: mps if available)")
    p.add_argument("--limit-batches", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    device = select_device(args.device)
    model, config = load_run(args.run_dir, device)
    _, _, test_loader = get_loaders(config["dataset"], config["batch_size"], config["seed"],
                                     config.get("val_size", 0))
    rows = evaluate_modes(model, test_loader, device, args.beams, args.limit_batches)
    print(format_table(rows))
    (Path(args.run_dir) / "eval.json").write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()
