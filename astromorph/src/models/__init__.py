from typing import Dict, Type
from torch import nn

from .fewer_layers_resnet import NLayerResnet

DEFAULT_MODELS: Dict[str, Type[nn.Module]] = {
    "n_layer_resnet": NLayerResnet,
}
