"""Load-balancing loss and routing stats (see CONTEXT.md)."""
import math

import torch


def router_usage(log_reach: torch.Tensor, log_route: torch.Tensor) -> torch.Tensor:
    """Reach-weighted mean child probability of one router, shape (k,)."""
    w = log_reach.exp()
    return (w[:, None] * log_route.exp()).sum(0) / w.sum().clamp(min=1e-12)


TARGET_MIN = 1e-3


def balance_loss(log_reach: dict, log_route: dict, targets: dict | None = None) -> torch.Tensor:
    """Sum over routers of mean_reach * KL(usage || target).

    `targets` maps a router to its children's target shares (see RouterGraph.targets);
    a router without one targets uniform. Target shares are clamped to at least
    TARGET_MIN and renormalised so the KL stays finite. Zero when usage equals the
    target; the reach factor means unreachable routers add nothing.
    """
    targets = targets or {}
    total = 0.0
    for name, lr in log_route.items():
        k = lr.shape[1]
        u = router_usage(log_reach[name], lr).clamp(min=1e-12)
        if name in targets:
            t = torch.as_tensor(targets[name], dtype=u.dtype, device=u.device).clamp(min=TARGET_MIN)
            kl = (u * (u.log() - (t / t.sum()).log())).sum()
        else:
            kl = (u * (u.log() + math.log(k))).sum()
        total = total + log_reach[name].exp().mean() * kl
    return total if torch.is_tensor(total) else torch.zeros(())


def routing_entropy(log_route: torch.Tensor) -> torch.Tensor:
    """Mean per-sample entropy of a router's child distribution."""
    return -(log_route.exp() * log_route).sum(1).mean()


def leaf_path_terms(log_reach: dict, leaf_log_probs: dict, y: torch.Tensor) -> dict:
    """Per-leaf, per-sample reach(leaf) * NLL(leaf, label), each shape (B,)."""
    return {n: log_reach[n].exp() * -lp.gather(1, y[:, None]).squeeze(1)
            for n, lp in leaf_log_probs.items()}


def path_loss(log_reach: dict, leaf_log_probs: dict, y: torch.Tensor) -> torch.Tensor:
    """Mean over samples of the sum over leaves of reach(leaf) * NLL(leaf, label)."""
    return sum(leaf_path_terms(log_reach, leaf_log_probs, y).values()).mean()
