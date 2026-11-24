"""Three models: a small CNN from scratch, frozen MobileNetV3 + new head, fully fine-tuned MobileNetV3."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class SmallCNN(nn.Module):
    """Four conv blocks (conv-BN-ReLU-pool), global average pooling, dropout, linear classifier."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def mobilenet(num_classes: int = 10, freeze_backbone: bool = False, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    return model


def build(name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    if name == "cnn-scratch":
        return SmallCNN(num_classes)
    if name == "mobilenet-frozen":
        return mobilenet(num_classes, freeze_backbone=True, pretrained=pretrained)
    if name == "mobilenet-finetune":
        return mobilenet(num_classes, freeze_backbone=False, pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
