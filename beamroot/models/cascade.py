"""Cascade model: every node owns a sized network reading the raw image (CONTEXT.md, ADR 0005)."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from beamroot.models.dag import DAGRouterModel, default_graph
from beamroot.models.graph import RouterGraph
from beamroot.models.stem import build_stem

STEPS = (1, 2, 3, 4)
SIZE_AXES = ("resolution", "hidden", "both")
LABEL_INIT_STD = 0.5  # label-leaf logit noise that breaks the routers' zero-gradient symmetry
SMALL_RES = 8  # at or below this resolution the stem pools once, not twice
ROUTER_HIDDEN = 64   # step 4 sizes equal the dag node sizes
LEAF_HIDDEN = 128
NEG = -1e4   # finite stand-in for log 0 where a gradient must stay defined


def size_ladder(step: int, axis: str, image_size: int, width: int, base_hidden: int):
    """(resolution, stem channels, hidden units) of a node at `step`; step 4 is the dag size."""
    if step not in STEPS:
        raise ValueError(f"size step must be in {STEPS}, got {step}")
    if axis not in SIZE_AXES:
        raise ValueError(f"size axis must be in {SIZE_AXES}, got {axis!r}")
    frac = step / STEPS[-1]
    res, chans, hidden = image_size, width, base_hidden
    if axis in ("resolution", "both"):
        res, chans = round(image_size * frac), max(1, round(width * frac))
    if axis in ("hidden", "both"):
        hidden = max(1, round(base_hidden * frac))
    return res, chans, hidden


def _avg_resize(x: torch.Tensor, res: int) -> torch.Tensor:
    """Adaptive average pool to res x res. MPS pooling needs sizes that divide evenly, so use
    the equivalent pooling matrix on each spatial axis."""
    n = x.shape[-1]
    idx = torch.arange(res)
    start, end = (idx * n) // res, -((-(idx + 1) * n) // res)
    cols = torch.arange(n)
    pool = ((cols[None] >= start[:, None]) & (cols[None] < end[:, None])).float()
    pool = (pool / pool.sum(1, keepdim=True)).to(x.device, x.dtype)
    return pool @ x @ pool.T


class NodeModel(nn.Module):
    """Downsample the image, own conv stem, hidden layer, output (router or classifier leaf)."""

    def __init__(self, kind: str, out_features: int, step: int, axis: str, in_channels: int,
                 image_size: int, width: int):
        super().__init__()
        base = ROUTER_HIDDEN if kind == "router" else LEAF_HIDDEN
        self.kind, self.step = kind, step
        self.res, chans, hidden = size_ladder(step, axis, image_size, width, base)
        self.stem = build_stem(in_channels, chans)
        if self.res <= SMALL_RES:
            self.stem = self.stem[:-1]  # a tiny image pooled to 1x1 leaves too few features
        with torch.no_grad():
            flat = self.stem(torch.zeros(1, in_channels, self.res, self.res)).numel()
        layers = [nn.Linear(flat, hidden), nn.ReLU()]
        if kind == "leaf":
            layers.append(nn.Dropout(0.25))
        self.net = nn.Sequential(*layers, nn.Linear(hidden, out_features))

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.res:
            x = _avg_resize(x, self.res)
        return self.stem(x).flatten(1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Activations after the ReLU, from the raw image."""
        return self.net[:2](self._flat(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(self._flat(x))
        return F.log_softmax(out, dim=1) if self.kind == "router" else out


class LabelLeaf(nn.Module):
    """A learned class distribution with no network: the same logits for every image."""

    def __init__(self, num_classes: int, step: int = 4):
        super().__init__()
        self.step = step
        # Near-uniform, not exactly: identical leaves give every router zero gradient.
        self.logits = nn.Parameter(torch.randn(num_classes) * LABEL_INIT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.to(x.device).expand(x.shape[0], -1)


class CascadeModel(DAGRouterModel):
    """Router graph whose nodes each read the image through their own sized network."""

    def __init__(self, graph: RouterGraph | None = None, in_channels: int = 1,
                 image_size: int = 28, num_classes: int = 10, width: int = 32,
                 size_axis: str = "both", start_step: int = 4,
                 steps: dict[str, int] | None = None, leaf: str = "classifier"):
        nn.Module.__init__(self)
        self.leaf = leaf
        self.prune_reach = 0.0   # training-time pruning threshold; set by train, not saved
        self.node_evals = self.rows_seen = 0   # node evaluations per image = ratio
        self.graph = graph if graph is not None else default_graph()
        self.size_axis = size_axis
        self.start_step = start_step
        self.steps = dict(steps) if steps is not None else {}
        if steps is None:  # root starts at start_step, children inherit their parent's
            for name in self.graph.topological_order():
                parents = self.graph.parents(name) if name != self.graph.root else []
                self.steps[name] = self.steps[parents[0]] if parents else start_step
        self.in_channels, self.image_size, self.width = in_channels, image_size, width
        self.num_classes = num_classes
        self.nodes = nn.ModuleDict()
        for name in self.graph.topological_order():
            self.nodes[name] = self.build_node(name)

    @property
    def config(self) -> dict:
        return {"in_channels": self.in_channels, "image_size": self.image_size,
                "num_classes": self.num_classes, "width": self.width,
                "size_axis": self.size_axis, "start_step": self.start_step, "leaf": self.leaf}

    def build_node(self, name: str) -> nn.Module:
        router = self.graph.nodes[name].kind == "router"
        if not router and self.leaf == "label":
            return LabelLeaf(self.num_classes, self.steps[name])
        out = len(self.graph.children(name)) if router else self.num_classes
        return NodeModel("router" if router else "leaf", out, self.steps[name], self.size_axis,
                         self.in_channels, self.image_size, self.width)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _run_rows(self, name: str, x: torch.Tensor, idx: torch.Tensor, fill: torch.Tensor):
        """Run node `name` on rows `idx` only; other rows take the finite `fill` vector."""
        out = fill.to(x.device).expand(x.shape[0], -1)
        if len(idx) == 0:
            return out
        sub = self.nodes[name](x[idx])
        if self.graph.nodes[name].kind == "leaf":
            sub = F.log_softmax(sub, dim=1)
        out = out.clone()   # index_copy is missing on MPS; index_put is differentiable
        out[idx] = sub
        return out

    def route(self, x: torch.Tensor) -> dict:
        """Like `DAGRouterModel.route`, but while training with `prune_reach` > 0 a node
        runs only on rows whose summed reach is at least that (pruned soft routing)."""
        self.rows_seen += x.shape[0]
        if not (self.training and self.prune_reach > 0):
            self.node_evals += x.shape[0] * len(self.graph.nodes)
            return super().route(x)
        log_r = torch.log(torch.tensor(self.prune_reach))
        order = self.graph.topological_order()
        incoming: dict[str, list[torch.Tensor]] = {n: [] for n in order}
        log_reach, log_route, leaf_log_probs = {}, {}, {}
        for name in order:
            if name == self.graph.root:
                log_reach[name] = x.new_zeros(x.shape[0])
                active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            else:
                # clamp keeps the all-pruned rows finite so logsumexp has no NaN gradient
                s = torch.logsumexp(torch.stack(incoming[name]).clamp(min=NEG), dim=0)
                active = s >= log_r.to(x.device)
                log_reach[name] = torch.where(active, s, torch.full_like(s, float("-inf")))
            idx = active.nonzero().squeeze(1)
            self.node_evals += len(idx)
            if self.graph.nodes[name].kind == "router":
                k = len(self.graph.children(name))
                log_route[name] = self._run_rows(name, x, idx, x.new_full((1, k), -math.log(k)))
                for i, child in enumerate(self.graph.children(name)):
                    incoming[child].append(log_reach[name] + log_route[name][:, i])
            else:
                c = self.num_classes
                leaf_log_probs[name] = self._run_rows(name, x, idx, x.new_full((1, c), -math.log(c)))
        mix = torch.stack([log_reach[n].clamp(min=NEG)[:, None] + leaf_log_probs[n]
                           for n in leaf_log_probs])
        return {"log_reach": log_reach, "log_route": log_route,
                "leaf_log_probs": leaf_log_probs, "log_probs": torch.logsumexp(mix, dim=0)}

    def checkpoint(self) -> dict:
        return {"kind": "cascade", "config": self.config, "graph": self.graph.to_dict(),
                "steps": dict(self.steps), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "CascadeModel":
        model = cls(graph=RouterGraph.from_dict(ckpt["graph"]), steps=ckpt["steps"],
                    **ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
