import argparse
import json
import time
from itertools import islice

import torch
import torch.nn.functional as F

from beamroot.data import dataset_info, get_loaders
from beamroot.device import select_device
from beamroot.growth import grow, merge_step, rebuild_optimizer, save_graph_snapshot
from beamroot.losses import balance_loss, path_loss, router_usage, routing_entropy
from beamroot.models.baseline import BaselineCNN, count_params, match_width
from beamroot.models.dag import DAGRouterModel
from beamroot.runs import create_run_dir, log_metrics


def evaluate(model, loader, device, limit_batches=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in islice(loader, limit_batches):
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def train_epoch(model, loader, opt, device, limit_batches=None, balance_weight=0.0, stats=None,
                parts=None):
    """Returns the list of per-step losses (cross-entropy + weighted balance loss).

    DAG models train on the path loss (ADR 0003); baselines on plain cross-entropy.
    For DAG models, `stats` (a dict) is filled with per-router mean usage and entropy,
    and `parts` (a dict) with the mean "ce" (path loss) and "balance" components of the loss.
    """
    model.train()
    losses = []
    sums, steps = {}, 0
    ce_sum = bal_sum = 0.0
    for x, y in islice(loader, limit_batches):
        opt.zero_grad()
        y = y.to(device)
        if isinstance(model, DAGRouterModel):
            out = model.route(x.to(device))
            ce = path_loss(out["log_reach"], out["leaf_log_probs"], y)
            loss, bal = ce, 0.0
            if balance_weight:
                bal = balance_weight * balance_loss(out["log_reach"], out["log_route"])
                loss = ce + bal
            ce_sum += ce.item()
            bal_sum += float(bal)
            with torch.no_grad():
                for name, lr in out["log_route"].items():
                    u = router_usage(out["log_reach"][name], lr).cpu()
                    e = routing_entropy(lr).cpu()
                    pu, pe = sums.get(name, (0, 0))
                    sums[name] = (pu + u, pe + e)
            steps += 1
        else:
            loss = F.cross_entropy(model(x.to(device)), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if stats is not None:
        for name, (u, e) in sums.items():
            stats[name] = {"usage": (u / steps).tolist(), "entropy": float(e / steps)}
    if parts is not None and steps:
        parts.update({"ce": ce_sum / steps, "balance": bal_sum / steps})
    return losses


def build_model(name: str, dataset: str = "mnist", width: int | None = None,
                hidden: int | None = None):
    info = dataset_info(dataset)
    width = width or info["width"]
    dims = {"in_channels": info["channels"], "image_size": info["image_size"],
            "num_classes": info["num_classes"], "width": width}
    if name == "baseline":
        return BaselineCNN(**dims, **({"hidden": hidden} if hidden else {}))
    if name == "dag":
        return DAGRouterModel(**dims)
    raise ValueError(f"unknown model {name!r}")


def matched_size(run_dir, dataset: str) -> tuple[int, int]:
    """Baseline (stem width, head hidden size) matching the DAG run's parameter count."""
    from beamroot.evaluate import load_run
    target = count_params(load_run(run_dir, "cpu")[0])
    info = dataset_info(dataset)
    return match_width(target, info["channels"], info["image_size"], info["num_classes"])


def save_payload(model):
    """DAG checkpoints carry graph structure + config; baseline stays a plain state_dict."""
    if isinstance(model, DAGRouterModel):
        return model.checkpoint()
    return model.state_dict()


def run(args) -> dict:
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    if args.merge and args.val_size == 0:
        raise SystemExit("--merge needs a validation set; use --val-size > 0")
    train_loader, val_loader, test_loader = get_loaders(
        args.dataset, args.batch_size, args.seed, args.val_size)
    if args.epochs is None:
        args.epochs = dataset_info(args.dataset)["epochs"]
    if args.match_params:
        if args.model != "baseline":
            raise SystemExit("--match-params only applies to --model baseline")
        args.width, args.hidden = matched_size(args.match_params, args.dataset)
    model = build_model(args.model, args.dataset, args.width, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {**vars(args), "resolved_device": str(device), "params": count_params(model)}
    run_dir = create_run_dir(args.name or f"{args.dataset}-{args.model}", args.runs_dir, config)
    growing = isinstance(model, DAGRouterModel) and args.grow_every > 0
    if growing:
        save_graph_snapshot(model, run_dir, 0)
    first_loss, last = None, {}
    fresh = []  # the latest split's new leaves; never merge candidates
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        stats, parts = {}, {}
        losses = train_epoch(model, train_loader, opt, device, args.limit_batches,
                             args.balance_weight, stats, parts)
        if first_loss is None:
            first_loss = losses[0]
        acc = evaluate(model, test_loader, device, args.limit_batches)
        last = {"epoch": epoch, "loss": sum(losses) / len(losses),
                "test_acc": acc, "seconds": time.time() - t0}
        if parts:
            last["ce_loss"], last["balance_loss"] = parts["ce"], parts["balance"]
        if stats:
            last["routing"] = stats
        if isinstance(model, DAGRouterModel):
            last["num_nodes"] = len(model.graph.nodes)
        log_metrics(run_dir, last)
        for name, s in stats.items():
            print(f"  {name}: usage {[round(v, 3) for v in s['usage']]} "
                  f"entropy {s['entropy']:.3f}", flush=True)
        print(f"epoch {epoch}: loss {last['loss']:.4f} test_acc {acc:.4f}", flush=True)
        # Grow after training on the current graph; never after the last epoch.
        if growing and epoch % args.grow_every == 0 and epoch < args.epochs:
            merges = []
            if args.merge:  # merge first so freed nodes count towards the split budget
                merges = merge_step(model, val_loader, device, epoch, run_dir,
                                    args.merge_threshold, limit_batches=args.growth_batches,
                                    protected=set(fresh))
                for m in merges:
                    print(f"  merge {m['merge'][1]} into {m['merge'][0]} (sym KL {m['kl']:.4f}), "
                          f"removed {m['removed']} ({m['num_nodes']} nodes)", flush=True)
                if not merges:
                    print(f"  merge: no pair below sym KL {args.merge_threshold}", flush=True)
            event = grow(model, train_loader, device, epoch, run_dir, args.max_nodes,
                         limit_batches=args.growth_batches, noise=args.split_noise)
            fresh = event["new_nodes"] if event else []
            if event or merges:
                opt = rebuild_optimizer(model, opt)
            if event:
                print(f"  split {event['split']} -> {event['new_nodes']} "
                      f"({event['num_nodes']} nodes)", flush=True)
                if "cluster_sizes" in event:
                    print(f"  cluster sizes {event['cluster_sizes']} router pre-train loss "
                          f"{event['router_pretrain_loss']} fallback {event['fallback']}",
                          flush=True)
    torch.save(save_payload(model), run_dir / "model.pt")
    if growing:  # growth changes the size; config.json records the final count
        config["params"] = count_params(model)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    if isinstance(model, DAGRouterModel) and not args.no_report:
        from beamroot.report import render_report
        for path in render_report(run_dir, device).values():
            print(f"wrote {path}", flush=True)
    return {"run_dir": run_dir, "first_loss": first_loss, **last}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "synthetic", "cifar10", "synthetic_cifar"])
    p.add_argument("--model", default="dag", choices=["dag", "baseline"])
    p.add_argument("--device", default=None, help="mps or cpu (default: mps if available)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="default depends on the dataset")
    p.add_argument("--width", type=int, default=None, help="stem width (default depends on the dataset)")
    p.add_argument("--hidden", type=int, default=None, help="baseline head hidden size (default 128)")
    p.add_argument("--match-params", default=None, metavar="RUN_DIR",
                   help="baseline only: size the CNN within 10%% of this DAG run's parameter count")
    p.add_argument("--val-size", type=int, default=5000,
                   help="training images held out (by --seed) as the validation set for merges")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--balance-weight", type=float, default=0.1,
                   help="load-balancing loss weight for dag models (0 disables)")
    p.add_argument("--grow-every", type=int, default=1,
                   help="dag only: split the worst leaf every K epochs (0 disables)")
    p.add_argument("--max-nodes", type=int, default=7,
                   help="dag only: stop growing when a split would exceed this many graph nodes")
    p.add_argument("--growth-batches", type=int, default=50,
                   help="training batches used to measure per-leaf loss and to cluster "
                        "the chosen leaf's images before a split")
    p.add_argument("--split-noise", type=float, default=1e-3,
                   help="std of the noise added to the kept units of a split's new leaves "
                        "(and to both copies when a split falls back to copy)")
    p.add_argument("--merge", action="store_true",
                   help="dag only: at each growth step, first merge node pairs whose "
                        "outputs on held-out data are near-identical")
    p.add_argument("--merge-threshold", type=float, default=0.05,
                   help="merge pairs whose mean symmetric KL is below this (nats)")
    p.add_argument("--no-report", action="store_true",
                   help="dag only: skip writing graph.png and report.html at the end of the run")
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--runs-dir", default="runs")
    return p.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
