"""Transfer-learning model: pretrained ResNet18, classifier replaced for 2 classes."""
import torch.nn as nn
from torchvision import models


def build_model(pretrained=True, freeze_backbone=False):
    """Return a ResNet18 for NORMAL vs PNEUMONIA.

    Args:
        pretrained: use ImageNet weights when True.
        freeze_backbone: freeze all layers except the final classifier
            (useful for a first, fast training run; unfreeze to fine-tune).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model
