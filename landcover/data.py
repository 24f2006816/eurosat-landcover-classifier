"""EuroSAT RGB: 27,000 Sentinel-2 satellite tiles (64x64 px) in 10 land-cover classes.

EuroSAT has no official test split, so we make a stratified 70/15/15 train/validation/test split
with a fixed seed. Validation picks the best epoch; the test set is touched once, at the end.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics, used by MobileNetV3


def train_transform(size: int) -> transforms.Compose:
    # Satellite tiles have no "up": flips and 90-degree rotations are label-preserving augmentations.
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(size: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def stratified_split(targets: list[int], fractions=(0.7, 0.15, 0.15), seed: int = 0):
    """Split indices so each class keeps the same proportions in train, validation and test."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(targets):
        by_class[t].append(i)
    gen = torch.Generator().manual_seed(seed)
    train, val, test = [], [], []
    for idx in by_class.values():
        order = [idx[j] for j in torch.randperm(len(idx), generator=gen).tolist()]
        n_train, n_val = int(fractions[0] * len(order)), int(fractions[1] * len(order))
        train += order[:n_train]
        val += order[n_train:n_train + n_val]
        test += order[n_train + n_val:]
    return sorted(train), sorted(val), sorted(test)


class _Transformed(Dataset):
    def __init__(self, base, indices, transform):
        self.subset = Subset(base, indices)
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        image, label = self.subset[i]
        return self.transform(image), label


def load_eurosat(root: Path = ROOT / "data"):
    from torchvision.datasets import EuroSAT

    return EuroSAT(root=str(root), download=True)


def make_loaders(base, size: int = 128, batch_size: int = 64, seed: int = 0, workers: int = 2):
    train_idx, val_idx, test_idx = stratified_split(list(base.targets), seed=seed)
    common = {"batch_size": batch_size, "num_workers": workers}
    return (
        DataLoader(_Transformed(base, train_idx, train_transform(size)), shuffle=True, **common),
        DataLoader(_Transformed(base, val_idx, eval_transform(size)), **common),
        DataLoader(_Transformed(base, test_idx, eval_transform(size)), **common),
    )
