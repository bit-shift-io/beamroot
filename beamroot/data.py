import torch
from torch.utils.data import DataLoader, TensorDataset


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
_VEHICLES = {"airplane", "automobile", "ship", "truck"}
CIFAR10_GROUPS = {i: ("vehicle" if c in _VEHICLES else "animal")
                  for i, c in enumerate(CIFAR10_CLASSES)}

# Per-dataset dimensions live here; train/evaluate/report read them via the run's config.
DATASETS = {
    "mnist": {"channels": 1, "image_size": 28, "num_classes": 10, "width": 32,
              "epochs": 5, "groups": None},
    "synthetic": {"channels": 1, "image_size": 28, "num_classes": 10, "width": 32,
                  "epochs": 5, "groups": None},
    "cifar10": {"channels": 3, "image_size": 32, "num_classes": 10, "width": 32,
                "epochs": 15, "groups": CIFAR10_GROUPS},
    "synthetic_cifar": {"channels": 3, "image_size": 32, "num_classes": 10, "width": 32,
                        "epochs": 5, "groups": CIFAR10_GROUPS},
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}")
    return DATASETS[name]


def synthetic_loaders(batch_size: int, n: int = 256, seed: int = 0,
                      channels: int = 1, size: int = 28):
    """Tiny learnable synthetic dataset (class = which quadrant is bright); no download."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 4, (n,), generator=g)
    x = torch.randn(n, channels, size, size, generator=g) * 0.1
    h = size // 2
    for i, c in enumerate(y.tolist()):
        r, col = (c // 2) * h, (c % 2) * h
        x[i, :, r:r + h, col:col + h] += 1.0
    ds = TensorDataset(x, y)
    gen = torch.Generator().manual_seed(seed)
    return (DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(ds, batch_size=batch_size))


def mnist_loaders(batch_size: int, root: str = "data", seed: int = 0):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    gen = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(test, batch_size=512))


def cifar10_loaders(batch_size: int, root: str = "data", seed: int = 0):
    """CIFAR-10; random crop (pad 4) and horizontal flip on the training set only."""
    from torchvision import datasets, transforms
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    gen = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(test, batch_size=512))


def get_loaders(dataset: str, batch_size: int, seed: int = 0):
    if dataset == "mnist":
        return mnist_loaders(batch_size, seed=seed)
    if dataset == "synthetic":
        return synthetic_loaders(batch_size, seed=seed)
    if dataset == "cifar10":
        return cifar10_loaders(batch_size, seed=seed)
    if dataset == "synthetic_cifar":
        return synthetic_loaders(batch_size, seed=seed, channels=3, size=32)
    raise ValueError(f"unknown dataset {dataset!r}")
