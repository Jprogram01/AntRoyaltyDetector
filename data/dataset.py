"""
PyTorch Dataset + DataLoader factory for queen/worker classification.
Handles class imbalance via WeightedRandomSampler.
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# Production label map (AntWeb caste data)
LABEL_MAP = {"queen": 1, "worker": 0}

# Hymenoptera baseline: ants=worker(0), bees treated as queen(1) for loop validation
BASELINE_LABEL_MAP = {"worker": 0, "bee": 1}

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

VAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class AntCasteDataset(Dataset):
    def __init__(self, root: Path, transform=None, baseline: bool = False):
        self.samples: list[tuple[Path, int]] = []
        label_map = BASELINE_LABEL_MAP if baseline else LABEL_MAP
        for caste, label in label_map.items():
            caste_dir = root / caste
            if not caste_dir.exists():
                continue
            for img_path in sorted(caste_dir.glob("*.jpg")):
                self.samples.append((img_path, label))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler (handles imbalance)."""
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        total = sum(counts.values())
        weights = {label: total / (count + 1e-6) for label, count in counts.items()}
        return torch.tensor([weights[label] for _, label in self.samples])


def make_loaders(
    raw_dir: Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 4,
    seed: int = 42,
    baseline: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split raw_dir into train/val/test loaders with balanced sampling on train."""
    full_dataset = AntCasteDataset(raw_dir, transform=None, baseline=baseline)
    n = len(full_dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Apply transforms via wrappers
    train_ds.dataset = AntCasteDataset(raw_dir, transform=TRAIN_TRANSFORMS, baseline=baseline)
    val_ds.dataset = AntCasteDataset(raw_dir, transform=VAL_TRANSFORMS, baseline=baseline)
    test_ds.dataset = AntCasteDataset(raw_dir, transform=VAL_TRANSFORMS, baseline=baseline)

    # Weighted sampler on train only
    all_weights = full_dataset.class_weights()
    train_weights = all_weights[train_ds.indices]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_ds),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
