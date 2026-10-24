"""Show the top-N paths beam search finds for test images of a trained DAG run.

    python -m beamroot.predict <run-dir> [--beam 3] [--samples 5] [--device cpu]

For each image: true label, beam prediction, then each retained leaf with its
highest-probability path, combined probability (all retained paths into the
leaf), and the leaf's predicted class.
"""
import argparse

import torch

from beamroot.beam import BeamResult, beam_search
from beamroot.data import get_loaders
from beamroot.device import select_device
from beamroot.evaluate import load_run


def format_result(index: int, label: int, res: BeamResult) -> str:
    lines = [f"sample {index}: true label {label}, beam prediction {res.pred} "
             f"({res.nodes_evaluated} nodes evaluated)"]
    for rank, p in enumerate(res.paths, 1):
        lines.append(f"  {rank}. {' -> '.join(p.nodes)}  prob {p.prob:.4f} "
                     f"(best path {p.path_prob:.4f})  class {p.pred}")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--device", default=None, help="mps or cpu (default: mps if available)")
    return p.parse_args(argv)


def main(argv=None) -> list[tuple[int, BeamResult]]:
    args = parse_args(argv)
    device = select_device(args.device)
    model, config = load_run(args.run_dir, device)
    _, _, test_loader = get_loaders(config["dataset"], config["batch_size"], config["seed"],
                                     config.get("val_size", 0))
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.extend(y.tolist())
        if len(ys) >= args.samples:
            break
    x = torch.cat(xs)[:args.samples].to(device)
    results = list(zip(ys[:args.samples], beam_search(model, x, args.beam)))
    for i, (label, res) in enumerate(results):
        print(format_result(i, label, res))
    return results


if __name__ == "__main__":
    main()
