"""Router graph structure (see CONTEXT.md, ADR 0001).

Pure structure, no weights: nodes are named, each holds a list of parents,
and edges are an ordered list of (parent, child) pairs. A router's children
are ordered by edge insertion; that order indexes its routing probabilities.
"""
from dataclasses import dataclass, field

KINDS = ("router", "leaf")


@dataclass
class Node:
    name: str
    kind: str
    parents: list[str] = field(default_factory=list)


class RouterGraph:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[tuple[str, str]] = []
        # Router -> balance target: its children's reach shares at split time, in
        # child order. Routers without one target uniform (see losses.balance_loss).
        self.targets: dict[str, list[float]] = {}

    def add_node(self, name: str, kind: str, parents: list[str] | None = None) -> Node:
        if name in self.nodes:
            raise ValueError(f"node {name!r} already exists")
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        if "." in name or not name:
            raise ValueError(f"invalid node name {name!r}")
        self.nodes[name] = Node(name, kind)
        for p in parents or []:
            self.add_edge(p, name)
        return self.nodes[name]

    def add_edge(self, parent: str, child: str) -> None:
        for n in (parent, child):
            if n not in self.nodes:
                raise KeyError(n)
        if self.nodes[parent].kind == "leaf":
            raise ValueError(f"leaf {parent!r} cannot have children")
        if (parent, child) in self.edges:
            raise ValueError(f"edge {parent!r} -> {child!r} already exists")
        if parent == child or parent in self.descendants(child):
            raise ValueError(f"edge {parent!r} -> {child!r} would create a cycle")
        self.edges.append((parent, child))
        self.nodes[child].parents.append(parent)

    def remove_node(self, name: str) -> None:
        if name not in self.nodes:
            raise KeyError(name)
        for p, c in [e for e in self.edges if name in e]:
            self.edges.remove((p, c))
            if c != name:
                self.nodes[c].parents.remove(p)
        del self.nodes[name]
        self.targets.pop(name, None)

    def split(self, name: str, k: int = 2) -> tuple[str, ...]:
        """Turn leaf `name` into a router with `k` new leaf children (see CONTEXT.md: Split).

        The node keeps its name and all its parents. Returns the new leaves' names.
        """
        if self.nodes[name].kind != "leaf":
            raise ValueError(f"only leaves can be split, {name!r} is a router")
        self.nodes[name].kind = "router"
        new = []
        for _ in range(k):
            new.append(self._fresh_name("leaf"))
            self.add_node(new[-1], "leaf", parents=[name])
        return tuple(new)

    def merge(self, keep: str, drop: str) -> list[str]:
        """Merge node `drop` into `keep` (see CONTEXT.md: Merge).

        Each edge parent -> drop is redirected to parent -> keep in place, so the
        parent's child order (= its router output columns) is unchanged. `drop`
        and any nodes left unreachable from the root are removed; returns the
        removed names. Nodes must be of the same kind, share no parent (that
        would need two output columns of one router to feed one node), and
        neither may be an ancestor of the other.
        """
        if keep == drop or self.nodes[keep].kind != self.nodes[drop].kind:
            raise ValueError(f"cannot merge {drop!r} into {keep!r}: need two nodes of one kind")
        if set(self.parents(keep)) & set(self.parents(drop)):
            raise ValueError(f"{keep!r} and {drop!r} share a parent")
        if drop in self.descendants(keep) or keep in self.descendants(drop):
            raise ValueError(f"{keep!r} and {drop!r} are ancestor and descendant")
        root, keep_desc = self.root, self.descendants(keep)
        for i, (p, c) in enumerate(self.edges):
            if c == drop:
                if p in keep_desc:  # unreachable given the checks above; guard the DAG anyway
                    raise ValueError(f"edge {p!r} -> {keep!r} would create a cycle")
                self.edges[i] = (p, keep)
                self.nodes[keep].parents.append(p)
        self.nodes[drop].parents.clear()
        return self.prune_unreachable(root)

    def prune_unreachable(self, root: str) -> list[str]:
        """Remove every node not reachable from `root`; returns the removed names.

        `root` is explicit because mid-edit the graph may have several parentless nodes.
        """
        reachable = {root} | self.descendants(root)
        removed = [n for n in self.nodes if n not in reachable]
        for n in removed:
            self.remove_node(n)
        return removed

    def _fresh_name(self, prefix: str) -> str:
        i = 0
        while f"{prefix}{i}" in self.nodes:
            i += 1
        return f"{prefix}{i}"

    def parents(self, name: str) -> list[str]:
        return list(self.nodes[name].parents)

    def children(self, name: str) -> list[str]:
        return [c for p, c in self.edges if p == name]

    def descendants(self, name: str) -> set[str]:
        seen, stack = set(), [name]
        while stack:
            for c in self.children(stack.pop()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    @property
    def root(self) -> str:
        roots = [n for n, node in self.nodes.items() if not node.parents]
        if len(roots) != 1:
            raise ValueError(f"graph must have exactly one root, found {roots}")
        return roots[0]

    def routers(self) -> list[str]:
        return [n for n in self.topological_order() if self.nodes[n].kind == "router"]

    def leaves(self) -> list[str]:
        return [n for n in self.topological_order() if self.nodes[n].kind == "leaf"]

    def topological_order(self) -> list[str]:
        """Kahn's algorithm; ties broken by insertion order so the result is deterministic."""
        indegree = {n: len(node.parents) for n, node in self.nodes.items()}
        ready = [n for n, d in indegree.items() if d == 0]
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for c in self.children(n):
                indegree[c] -= 1
                if indegree[c] == 0:
                    ready.append(c)
        return order

    def to_dict(self) -> dict:
        return {"nodes": [{"name": n.name, "kind": n.kind} for n in self.nodes.values()],
                "edges": [list(e) for e in self.edges],
                "targets": {n: list(t) for n, t in self.targets.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "RouterGraph":
        g = cls()
        for n in d["nodes"]:
            g.add_node(n["name"], n["kind"])
        for p, c in d["edges"]:
            g.add_edge(p, c)
        g.targets = {n: list(t) for n, t in d.get("targets", {}).items()}  # optional: older runs
        return g
