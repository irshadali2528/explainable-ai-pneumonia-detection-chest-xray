"""Datasets and data loaders for chest X-ray pneumonia classification.

Expected layout (see README):
    data/train/NORMAL, data/train/PNEUMONIA
    data/val/NORMAL,   data/val/PNEUMONIA
    data/test/NORMAL,  data/test/PNEUMONIA
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ["NORMAL", "PNEUMONIA"]


def get_transforms(train: bool = False):
    """224x224, 3-channel, ImageNet normalisation. Training adds augmentation."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """Return train/val/test loaders and the class names."""
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(data_dir / "val", transform=get_transforms(train=False))
    test_ds = datasets.ImageFolder(data_dir / "test", transform=get_transforms(train=False))

    # ImageFolder sorts classes alphabetically: NORMAL (0), PNEUMONIA (1).
    assert train_ds.classes == CLASSES, f"Expected {CLASSES}, got {train_ds.classes}"

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, train_ds.classes
