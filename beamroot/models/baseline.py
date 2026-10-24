import math

import torch
from torch import nn

from beamroot.models.stem import build_stem


class BaselineCNN(nn.Module):
    """Plain CNN: conv stem followed by an MLP classifier head."""

    def __init__(self, in_channels: int = 1, image_size: int = 28,
                 num_classes: int = 10, width: int = 32, hidden: int = 128):
        super().__init__()
        self.stem = build_stem(in_channels, width)
        feat = width * 2 * (image_size // 4) ** 2
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(feat, hidden), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stem(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def match_width(target: int, in_channels: int, image_size: int, num_classes: int,
                max_width: int = 1024, default_hidden: int = 128) -> tuple[int, int]:
    """(stem width, head hidden size) whose BaselineCNN parameter count is closest to `target`.

    Stem width alone is too coarse at small sizes, so the head's hidden size is solved
    (the count is linear in it) for each width; among widths that land within 1% of the
    target, the one whose hidden size is nearest the default keeps the architecture's shape.
    Parameter count grows with width, so the scan stops once even hidden=1 overshoots.
    """
    def count(w, h):
        return count_params(BaselineCNN(in_channels, image_size, num_classes, w, h))

    best, best_gap, shaped, shaped_dist = (1, 1), None, None, None
    for width in range(1, max_width + 1):
        p1 = count(width, 1)
        slope = count(width, 2) - p1
        hidden = max(1, round(1 + (target - p1) / slope))
        gap = abs(count(width, hidden) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = (width, hidden), gap
        dist = abs(math.log(hidden / default_hidden))
        if gap <= 0.01 * target and (shaped is None or dist < shaped_dist):
            shaped, shaped_dist = (width, hidden), dist
        if p1 > target:
            break
    return shaped or best
