from typing import List, Dict
import pydantic
import torch
from torch import nn
from torchvision.models import resnet18


class LayerSettings(pydantic.BaseModel):
    """Settings to be used for constructing a reduced ResNet18 network.

    Attributes:
        excluded_layers: which layers to exclude from the network
        embedding_dim: dimension of features going into the final linear layer
    """

    excluded_layers: List[str]
    embedding_dim: int


class NLayerResnet(nn.Module):
    """A neural network based on ResNet18.

    This network uses the first n layers of the ResNet18, with the intent to
    find large-scale patterns instead of small features in the image.
    The last layer is a torch.nn.Linear layer that outputs to 64 features.

    Attributes:
        LAST_LAYER_SETTINGS: settings corresponding to the last convolution layer
    """

    LAST_LAYER_SETTINGS: Dict[str, LayerSettings] = {
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
        """Create a reduced ResNet18-based neural network.

        Args:
            last_layer: the last convolutional layer of the network
        """
        super().__init__()
        settings = self.LAST_LAYER_SETTINGS[last_layer]
        # The input_features of the last layer (fc) depend on the final
        # convolutional layer, which is why we remove "fc" as well
        excluded_layers = settings.excluded_layers + ["fc"]

        basemodel = resnet18()
        for name, child in basemodel.named_children():
            if name not in excluded_layers:
                self.add_module(name, child)

        self.add_module(
            "fc", nn.Linear(in_features=settings.embedding_dim, out_features=64)
        )
        self.embedding_dim = settings.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward a torch.Tensor through the network.

        Args:
            x: input tensor

        Returns:
            an embedding
        """
        for name, child in self.named_children():
            # Linear layer fc only accepts a flattened tensor
            if name != "fc":
                x = child(x)
            else:
                x = child(torch.flatten(x, start_dim=1))

        return x
