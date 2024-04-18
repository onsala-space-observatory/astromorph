from typing import Dict, Type
from torch import nn

from .fewer_layers_resnet import NLayerResnet
from .cloudscanner import CloudScanner

DEFAULT_MODELS: Dict[str, Type[nn.Module]] = {
    "n_layer_resnet": NLayerResnet,
    "cloudscanner": CloudScanner
}
