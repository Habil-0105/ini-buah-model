"""Backbone builders (torchvision)."""
import torch.nn as nn
from torchvision import models

ARCHS = ("mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0")


def build_model(arch, num_classes, pretrained=True):
    if arch == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}; choose from {ARCHS}")
    return m


def backbone_head(model):
    return model.features, model.classifier
