from torch import nn


def build_stem(in_channels: int = 1, width: int = 32) -> nn.Sequential:
    """Shared conv stem (see CONTEXT.md). Halves spatial size twice."""
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(width, width * 2, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
