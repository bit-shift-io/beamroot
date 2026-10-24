import torch


def select_device(requested: str | None = None) -> torch.device:
    """Return the requested device, or MPS if available, else CPU."""
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
