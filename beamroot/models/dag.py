"""Stem + Router graph with soft routing (see CONTEXT.md, ADR 0001)."""
import torch
import torch.nn.functional as F
from torch import nn

from beamroot.models.graph import RouterGraph
from beamroot.models.nodes import Leaf, RouterNode
from beamroot.models.stem import build_stem


def default_graph() -> RouterGraph:
    """Root router with two leaves."""
    g = RouterGraph()
    g.add_node("root", "router")
    g.add_node("leaf0", "leaf", parents=["root"])
    g.add_node("leaf1", "leaf", parents=["root"])
    return g


class DAGRouterModel(nn.Module):
    """Images -> Stem -> Router graph -> reach-weighted mix of Leaf predictions.

    `forward` returns log-probabilities over classes (so `F.cross_entropy`
    and `argmax` both work on it directly).
    """

    def __init__(self, graph: RouterGraph | None = None, in_channels: int = 1,
                 image_size: int = 28, num_classes: int = 10, width: int = 32):
        super().__init__()
        self.config = {"in_channels": in_channels, "image_size": image_size,
                       "num_classes": num_classes, "width": width}
        self.graph = graph if graph is not None else default_graph()
        self.stem = build_stem(in_channels, width)
        self.in_features = width * 2 * (image_size // 4) ** 2
        self.num_classes = num_classes
        # ModuleDict so nodes added later reach the optimiser and state_dict.
        self.nodes = nn.ModuleDict()
        for name in self.graph.topological_order():
            self.nodes[name] = self.build_node(name)

    def build_node(self, name: str) -> nn.Module:
        """Fresh module for a graph node, sized to its current children."""
        if self.graph.nodes[name].kind == "router":
            return RouterNode(self.in_features, len(self.graph.children(name)))
        return Leaf(self.in_features, self.num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """What every node reads: flattened shared-stem features."""
        return self.stem(x).flatten(1)

    def route(self, x: torch.Tensor) -> dict:
        """Soft forward over the whole graph, evaluating each node once.

        Returns:
          log_reach:      {node: (B,)} log of the summed probability of all paths to it
          log_route:      {router: (B, num_children)} log routing probs, ordered as graph.children
          leaf_log_probs: {leaf: (B, C)} each leaf's log class distribution
          log_probs:      (B, C) log of the reach-weighted leaf mixture
        """
        feats = self.features(x)
        order = self.graph.topological_order()
        incoming: dict[str, list[torch.Tensor]] = {n: [] for n in order}
        log_reach, log_route, leaf_log_probs = {}, {}, {}
        for name in order:
            if name == self.graph.root:
                log_reach[name] = feats.new_zeros(feats.shape[0])
            else:
                log_reach[name] = torch.logsumexp(torch.stack(incoming[name]), dim=0)
            if self.graph.nodes[name].kind == "router":
                log_route[name] = self.nodes[name](feats)
                for i, child in enumerate(self.graph.children(name)):
                    incoming[child].append(log_reach[name] + log_route[name][:, i])
            else:
                leaf_log_probs[name] = F.log_softmax(self.nodes[name](feats), dim=1)
        leaves = list(leaf_log_probs)
        mix = torch.stack([log_reach[n][:, None] + leaf_log_probs[n] for n in leaves])
        return {"log_reach": log_reach, "log_route": log_route,
                "leaf_log_probs": leaf_log_probs, "log_probs": torch.logsumexp(mix, dim=0)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.route(x)["log_probs"]

    def checkpoint(self) -> dict:
        return {"config": self.config, "graph": self.graph.to_dict(),
                "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "DAGRouterModel":
        model = cls(graph=RouterGraph.from_dict(ckpt["graph"]), **ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
