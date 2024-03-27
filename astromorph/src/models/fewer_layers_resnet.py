from typing import List
import pydantic
import torch
from torch import nn
from torchvision.models import resnet18


class LayerSettings(pydantic.BaseModel):
    excluded_layers: List[str]
    embedding_dim: int


class NLayerResnet(nn.Module):
    LAST_LAYER_SETTINGS = {
        "layer1": LayerSettings(
            excluded_layers=["layer2", "layer3", "layer4"], embedding_dim=64
        ),
        "layer2": LayerSettings(
            excluded_layers=["layer3", "layer4"], embedding_dim=128
        ),
        "layer3": LayerSettings(excluded_layers=["layer4"], embedding_dim=256),
        "layer4": LayerSettings(excluded_layers=[], embedding_dim=512),
    }

    def __init__(self, last_layer: str = "layer4") -> None:
        super().__init__()
        settings = self.LAST_LAYER_SETTINGS[last_layer]
        excluded_layers = settings.excluded_layers + ["fc"]
        embedding_dim = settings.embedding_dim

        basemodel = resnet18()
        for name, child in basemodel.named_children():
            if name not in excluded_layers:
                self.add_module(name, child)

        self.add_module("fc", nn.Linear(in_features=embedding_dim, out_features=64))

    def forward(self, x):
        for name, child in self.named_children():
            if name != "fc":
                x = child(x)
            else:
                x = child(torch.flatten(x, 1))

        return x
