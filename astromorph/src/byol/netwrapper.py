from typing import Union

import torch
from torch import nn
from loguru import logger

from .mlp import MultiLayerPerceptron

class NetWrapper(nn.Module):
    """Class to wrap around another neural network.

    This class takes care of some of the admin and overhead when intercepting
    model output.

    Attributes:
        network: neural network to be wrapped
        layer: which layer of the network to intercept
        hidden: dictionary to store intercepted output
        projector: an MLP to do projection of the embeddings
    """
    def __init__(
        self,
        network: nn.Module,
        representation_size: int,
        layer: Union[str, int] = -2,
        projection_size: int = 256,
        projection_hidden_size: int = 1024,
    ) -> None:
        """Init the NetWrapper.

        Args:
            network: neural network to be wrapped
            representation_size: size of the embeddings intercepted from network
            layer: which layer of the neural network to intercept
            projection_size: final projection size
            projection_hidden_size: size of hidden layer in projector
        """
        super().__init__()

        self.network: nn.Module = network
        self.layer: Union[str, int] = layer

        # Variable to store the data emitted by the hiddenn layer in the network
        self.hidden = {}
        # Register forward hook at the right layer
        self._register_hook()

        self.projector = MultiLayerPerceptron(representation_size, projection_hidden_size, projection_size)

    def _find_layer(self) -> nn.Module:
        """Find layer that will be intercepted

        Returns:
            a layer of a network
        """
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
        """Hook function to emit output to self.hidden via a forward hook.

        Args:
            model: layer to be intercepted
            input: the input that goes into the layer
            output: output emitted by the layer
        """
        # Get the device name
        device = input[0].device
        # Store the output based on device name
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        """Register the _hook function with the layer we want to intercept."""
        layer = self._find_layer()
        layer.register_forward_hook(self._hook)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get the embedding representation of an image.

        Args:
            x: input image

        Returns:
            image embedding
        """
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
