"""Torchvision ImageNet model factories for CUB transfer learning."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B3_Weights,
    ResNet50_Weights,
    efficientnet_b3,
    resnet50,
)


MODEL_SPECS = {
    "resnet50": (ResNet50_Weights.DEFAULT, 224),
    "efficientnet_b3": (EfficientNet_B3_Weights.DEFAULT, 300),
}


def create_transfer_model(model_name: str, num_classes: int = 200, pretrained: bool = True):
    """Return a pretrained model, its weights enum, and its recommended crop size."""
    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.DEFAULT
        model = efficientnet_b3(weights=weights if pretrained else None)
        final = model.classifier[-1]
        model.classifier[-1] = nn.Linear(final.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model {model_name!r}; choose from {sorted(MODEL_SPECS)}.")
    return model, weights


def configure_trainable_layers(model, model_name: str, stage: str, blocks: int = 1) -> None:
    """Freeze all pretrained layers for extraction; unfreeze the top blocks for tuning."""
    if stage not in {"feature_extraction", "fine_tuning"}:
        raise ValueError("stage must be 'feature_extraction' or 'fine_tuning'.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model_name == "resnet50":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        if stage == "fine_tuning":
            for layer_name in ({1: ("layer4",), 2: ("layer3", "layer4")}).get(
                max(1, min(2, int(blocks))), ("layer4",)
            ):
                for parameter in getattr(model, layer_name).parameters():
                    parameter.requires_grad = True
    else:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        if stage == "fine_tuning":
            features = list(model.features.children())
            count = max(1, min(len(features), int(blocks)))
            for block in features[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
