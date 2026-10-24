"""Router node and Leaf heads (see CONTEXT.md).

Both read flattened Stem features, never a parent's output, so a node's
behaviour does not depend on which path reached it.
"""
import torch
import torch.nn.functional as F
from torch import nn


class RouterNode(nn.Module):
    """Small MLP giving a log-probability per child."""

    def __init__(self, in_features: int, num_children: int, hidden: int = 64):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, num_children)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.out(self.hidden(feats)), dim=1)


class Leaf(nn.Module):
    """Classifier head giving logits over the dataset's classes."""

    def __init__(self, in_features: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)
