import torch
from torch import nn
from torchvision.models import resnet18


class NLayerResnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        basemodel = resnet18()
        excluded_layers = ["layer2", "layer3", "layer4", "fc"]

        for name, child in basemodel.named_children():
            if name not in excluded_layers:
                self.add_module(name, child)

        self.add_module("fc", nn.Linear(in_features=64, out_features=64))

    def forward(self, x):
        for name, child in self.named_children():
            if name != "fc":
                x = child(x)
            else:
                x = child(torch.flatten(x, 1))

        return x
