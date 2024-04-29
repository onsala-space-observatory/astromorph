from typing import Union

import torch
from torch import nn
from loguru import logger

from .mlp import MultiLayerPerceptron

class NetWrapper(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        representation_size: int,
        layer: Union[str, int] = -2,
        projection_size: int = 256,
        projection_hidden_size: int = 1024,
    ) -> None:
        super().__init__()

        self.network: nn.Module = network
        self.layer: Union[str, int] = layer

        # Variable to store the data emitted by the hiddenn layer in the network
        self.hidden = {}
        self._register_hook()

        self.projector = MultiLayerPerceptron(representation_size, projection_hidden_size, projection_size)

    def _find_layer(self) -> nn.Module:
        try:
            if isinstance(self.layer, int):
                children = list(self.network.children())
                return children[self.layer]
            elif isinstance(self.layer, str):
                modules = dict(list(self.network.named_modules()))
                return modules[self.layer]
        except KeyError:
            logger.error("Layer {} not found in model", self.layer)
            raise SystemExit

    def _hook(self, model: nn.Module, input: torch.Tensor, output: torch.Tensor):
        """Function to emit output to self.hidden via a forward hook.

        Args:
            model:
            input:
            output:
        """
        # Get the device name
        device = input[0].device
        # store the output based on device name
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        """Register the _hook function with the layer we want to intercept."""
        layer = self._find_layer()
        layer.register_forward_hook(self._hook)

    def get_representation(self, x: torch.Tensor):
        # Ensure the hidden dict is clear, to not have previous runs contaminate our output
        self.hidden.clear()
        _ = self.network(x)
        output = self.hidden[x.device]
        self.hidden.clear()

        if output is None:
            logger.error("Layer {} never emitted any output", self.layer)
        else:
            return output

    def forward(self, x: torch.Tensor, return_projection: bool = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation
        
        projection = self.projector(representation)

        return projection, representation
