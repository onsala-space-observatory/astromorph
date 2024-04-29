import torch
from torch import nn


class CloudScanner(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-5),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-5),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=128, out_features=64)

    def forward(self, x):
        for name, child in self.named_children():
            if name != "fc":
                x = child(x)
            else:
                x = child(torch.flatten(x, start_dim=1))
