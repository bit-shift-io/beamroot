"""Inference cost: multiply-accumulates (MACs) from conv and linear shapes."""
import torch
from torch import nn


def module_macs(module: nn.Module, *inputs: torch.Tensor) -> int:
    """MACs of one forward pass of `module` on `inputs` (per image of the batch)."""
    total = [0]

    def hook(m, _inp, out):
        batch = out.shape[0]
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
            total[0] += out.numel() // batch * k
        else:
            total[0] += m.in_features * m.out_features

    handles = [m.register_forward_hook(hook) for m in module.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            module(*inputs)
    finally:
        module.train(was_training)
        for h in handles:
            h.remove()
    return total[0]


def model_costs(model) -> tuple[int, dict[str, int]]:
    """(shared stem MACs, {node: MACs}). Cascade models have no shared stem (0)."""
    cfg = getattr(model, "config", None) or {}
    chans = cfg.get("in_channels", getattr(model, "in_channels", 1))
    size = cfg.get("image_size", getattr(model, "image_size", 28))
    device = next(model.parameters()).device
    img = torch.zeros(1, chans, size, size, device=device)
    stem = getattr(model, "stem", None)
    stem_macs = module_macs(stem, img) if stem is not None else 0
    with torch.no_grad():
        feats = model.features(img)
    return stem_macs, {name: module_macs(node, feats) for name, node in model.nodes.items()}
