"""Tree model: a tree grown inside one plain conv network (CONTEXT.md, ADR 0006)."""
import torch
import torch.nn.functional as F
from torch import nn

from beamroot.models.cascade import _avg_resize
from beamroot.models.graph import RouterGraph

def _conv(i: int, o: int) -> list[nn.Module]:
    return [nn.Conv2d(i, o, 3, padding=1), nn.ReLU()]


CONVS = (1, 1, 1, 2)  # convs per `deep4` block


def deep4_widths(width: int = 16) -> list[int]:
    """Output channels of each conv of the full `deep4` network, in order."""
    return [width, width * 2, width * 4, width * 8, width * 8]


def deep4_segment(in_channels: int, start: int, end: int, widths: list[int]
                  ) -> list[nn.Sequential]:
    """`deep4` blocks `start` to `end - 1` reading `in_channels`, with `widths` the output
    channels of each of their convs, in order (no BatchNorm: convs carry every
    per-channel parameter)."""
    blocks, i = [], 0
    for b in range(start, end):
        layers = []
        for _ in range(CONVS[b]):
            layers += _conv(in_channels, widths[i])
            in_channels, i = widths[i], i + 1
        blocks.append(nn.Sequential(*layers, nn.MaxPool2d(2)))
    return blocks


def deep4_blocks(in_channels: int, width: int = 16) -> list[nn.Sequential]:
    """The `deep4` conv blocks of experiment 0007: width x (1, 2, 4, 8) channels, each block
    halving the spatial size; the last block has two convs. Width 16 gives 16/32/64/128."""
    return deep4_segment(in_channels, 0, len(CONVS), deep4_widths(width))


class Classifier(nn.Module):
    """Leaf head: global average pool, then a linear layer giving class logits."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.mean(dim=(2, 3)))  # mean, not adaptive pool: safe on MPS


class TreeNode(nn.Module):
    """A tree node: its segment (the conv blocks it owns) and a head reading the segment's
    output. `forward(input)` is head(segment(input)): class logits for a leaf."""

    def __init__(self, segment: nn.Sequential, head: nn.Module):
        super().__init__()
        self.segment, self.head = segment, head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.segment(x))


class NodeCache:
    """Per-batch memo of node inputs and outputs, so each segment runs once per batch.

    `input(name)` is the root's image or the parent's segment output; `output(name)` is the
    node's head output on it.
    """

    def __init__(self, model: "TreeModel", x: torch.Tensor):
        self.model, self.x = model, x
        self._seg: dict[str, torch.Tensor] = {}
        self._out: dict[str, torch.Tensor] = {}

    def input(self, name: str) -> torch.Tensor:
        parents = self.model.graph.parents(name)
        return self._segment_out(parents[0]) if parents else self.x

    def _segment_out(self, name: str) -> torch.Tensor:
        if name not in self._seg:
            self._seg[name] = self.model.nodes[name].segment(self.input(name))
        return self._seg[name]

    def output(self, name: str) -> torch.Tensor:
        if name not in self._out:
            self._out[name] = self.model.nodes[name].head(self._segment_out(name))
        return self._out[name]


BLOCKS = ("b1", "b2", "b3", "b4")  # names of the `deep4` blocks, in order


class _Pool(nn.Module):
    """Adaptive average pool to `side` x `side`; sizes that do not divide evenly use the
    pooling-matrix form, which MPS supports."""

    def __init__(self, side: int):
        super().__init__()
        self.side = side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.side == 0:
            return F.adaptive_avg_pool2d(x, self.side)
        return _avg_resize(x, self.side)


class RouterHead(nn.Module):
    """Tree router head (experiment 0007's `Head`): average-pool the split block's output to
    at most 4x4, then Linear 64, ReLU, Linear `out`. Returns log routing probabilities."""

    def __init__(self, channels: int, size: int, out: int):
        super().__init__()
        side = min(4, size)
        self.pool = _Pool(side)
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(channels * side * side, 64), nn.ReLU(),
                                 nn.Linear(64, out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.net(self.pool(x)), dim=1)


class TreeModel(nn.Module):
    """Tree of nodes inside the `deep4` network; one leaf (the root) is the plain network.

    Node interface (what beam search, `route` and cost use):
      - `nodes[name]` is a `TreeNode`; `nodes[name](inp)` gives the node's output: class
        logits for a leaf, log routing probabilities (child order) for a router.
      - A node's input is its parent's output at the parent's split block, i.e. the parent's
        segment output. The root's input is the image.
      - `node_cache(x)` returns a `NodeCache` for batch `x`: `.input(name)` and
        `.output(name)`, each segment evaluated at most once per batch. Callers reach node
        inputs through it, never by recomputing ancestors.

    Split bookkeeping: `split_blocks[router]` is the router's split block (a name in
    `BLOCKS`) and `groups[router]` its children's class groups, in child order. A node's
    segment starts after its parent's split block and ends at its own split block (router)
    or the last block (leaf). Built by `growth.tree_split`.

    Branch widths: `channels[node]` lists the output channels of each conv of the node's
    segment, for nodes narrowed by branch keep; other nodes have `deep4`'s widths
    (`node_widths`). A node's segment reads its parent's last conv width.

    `forward` returns log-probabilities over classes, like `DAGRouterModel`.
    """

    def __init__(self, graph: RouterGraph | None = None, in_channels: int = 3,
                 image_size: int = 32, num_classes: int = 10, width: int = 16,
                 split_blocks: dict[str, str] | None = None,
                 groups: dict[str, list[list[int]]] | None = None,
                 channels: dict[str, list[int]] | None = None):
        super().__init__()
        self.config = {"in_channels": in_channels, "image_size": image_size,
                       "num_classes": num_classes, "width": width}
        self.num_classes = num_classes
        if graph is None:
            graph = RouterGraph()
            graph.add_node("root", "leaf")
        self.graph = graph
        self.split_blocks = dict(split_blocks or {})
        self.groups = {r: [list(g) for g in gs] for r, gs in (groups or {}).items()}
        self.channels = {n: list(w) for n, w in (channels or {}).items()}
        self.nodes = nn.ModuleDict()
        for name in self.graph.topological_order():
            self.nodes[name] = self.build_node(name)

    def start_block(self, name: str) -> int:
        """Index in `BLOCKS` of the first block node `name`'s segment owns."""
        parents = self.graph.parents(name)
        return BLOCKS.index(self.split_blocks[parents[0]]) + 1 if parents else 0

    def node_classes(self, name: str) -> list[int]:
        """Classes whose true path reaches `name`: all for the root, else its group."""
        parents = self.graph.parents(name)
        if not parents:
            return list(range(self.num_classes))
        return self.groups[parents[0]][self.graph.children(parents[0]).index(name)]

    def end_block(self, name: str) -> int:
        """Index in `BLOCKS` one past the last block node `name`'s segment owns."""
        if self.graph.nodes[name].kind == "leaf":
            return len(BLOCKS)
        return BLOCKS.index(self.split_blocks[name]) + 1

    def default_widths(self, name: str) -> list[int]:
        """`deep4`'s conv widths over node `name`'s segment."""
        lo = sum(CONVS[:self.start_block(name)])
        return deep4_widths(self.config["width"])[lo:sum(CONVS[:self.end_block(name)])]

    def node_widths(self, name: str) -> list[int]:
        """Output channels of each conv of node `name`'s segment, in order."""
        return list(self.channels.get(name, self.default_widths(name)))

    def set_widths(self, name: str, widths: list[int]) -> None:
        """Record node `name`'s conv widths (kept only where they differ from `deep4`'s)."""
        if list(widths) == self.default_widths(name):
            self.channels.pop(name, None)
        else:
            self.channels[name] = list(widths)

    def in_width(self, name: str) -> int:
        """Channels of node `name`'s input: the image's, or its parent's last conv's."""
        parents = self.graph.parents(name)
        return self.node_widths(parents[0])[-1] if parents else self.config["in_channels"]

    def out_width(self, name: str) -> int:
        """Channels of node `name`'s segment output (its input's if it owns no blocks)."""
        widths = self.node_widths(name)
        return widths[-1] if widths else self.in_width(name)

    def block_width(self, name: str, block: int) -> int:
        """Output channels of block index `block` of node `name`'s segment."""
        return self.node_widths(name)[sum(CONVS[self.start_block(name):block + 1]) - 1]

    def build_node(self, name: str) -> TreeNode:
        """A freshly initialised node: its segment of `deep4`'s blocks at its widths, then
        a `Classifier` (leaf) or a `RouterHead` on its split block's output (router)."""
        cfg = self.config
        start, end = self.start_block(name), self.end_block(name)
        segment = nn.Sequential(*deep4_segment(self.in_width(name), start, end,
                                               self.node_widths(name)))
        if self.graph.nodes[name].kind == "leaf":
            return TreeNode(segment, Classifier(self.out_width(name), cfg["num_classes"]))
        head = RouterHead(self.out_width(name), cfg["image_size"] // 2 ** end,
                          len(self.graph.children(name)))
        return TreeNode(segment, head)

    def node_cache(self, x: torch.Tensor) -> NodeCache:
        return NodeCache(self, x)

    def route(self, x: torch.Tensor) -> dict:
        """Forward over the tree. Returns `log_reach`, `log_route`, `leaf_log_probs` and
        `log_probs` as `DAGRouterModel.route` does: soft routing, every node on every
        image. Training uses `teacher_forced_loss` instead."""
        cache = self.node_cache(x)
        log_reach, log_route, leaf_log_probs = {}, {}, {}
        for name in self.graph.topological_order():
            parents = self.graph.parents(name)
            if not parents:
                log_reach[name] = x.new_zeros(x.shape[0])
            else:
                i = self.graph.children(parents[0]).index(name)
                log_reach[name] = log_reach[parents[0]] + log_route[parents[0]][:, i]
            if self.graph.nodes[name].kind == "router":
                log_route[name] = cache.output(name)
            else:
                leaf_log_probs[name] = F.log_softmax(cache.output(name), dim=1)
        mix = torch.stack([log_reach[n][:, None] + lp for n, lp in leaf_log_probs.items()])
        return {"log_reach": log_reach, "log_route": log_route,
                "leaf_log_probs": leaf_log_probs, "log_probs": torch.logsumexp(mix, dim=0)}

    def teacher_forced_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training loss (CONTEXT.md: Teacher forcing).

        Each image follows its true path: from the root, at each router, to the child whose
        group holds its class. Along that path:
          - each router it passes adds the cross-entropy of the router's output against the
            image's group label (the index of the child group holding its class);
          - the leaf it ends at adds the cross-entropy of the leaf's logits against its class.
        A node's segment runs only on the images whose true path reaches it, so a node gets
        no gradient from images of other groups. The loss is the summed terms divided by the
        batch size: per image, the sum of its path's cross-entropies, averaged over images.
        Nodes no image of the batch reaches add nothing.
        """
        total = x.new_zeros(())

        def visit(name: str, inp: torch.Tensor, labels: torch.Tensor) -> None:
            nonlocal total
            node = self.nodes[name]
            feats = node.segment(inp)
            out = node.head(feats)
            children = self.graph.children(name)
            if not children:
                total = total + F.cross_entropy(out, labels, reduction="sum")
                return
            lookup = torch.full((self.num_classes,), -1, dtype=torch.long, device=labels.device)
            for i, group in enumerate(self.groups[name]):
                lookup[group] = i
            group_of = lookup[labels]
            total = total + F.nll_loss(out, group_of, reduction="sum")
            for i, child in enumerate(children):
                rows = group_of == i
                if rows.any():
                    visit(child, feats[rows], labels[rows])

        visit(self.graph.root, x, y)
        return total / x.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.route(x)["log_probs"]

    def checkpoint(self) -> dict:
        return {"kind": "tree", "config": self.config, "graph": self.graph.to_dict(),
                "split_blocks": dict(self.split_blocks), "groups": self.groups,
                "channels": dict(self.channels), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "TreeModel":
        model = cls(graph=RouterGraph.from_dict(ckpt["graph"]), **ckpt["config"],
                    split_blocks=ckpt.get("split_blocks"), groups=ckpt.get("groups"),
                    channels=ckpt.get("channels"))
        model.load_state_dict(ckpt["state_dict"])
        return model
