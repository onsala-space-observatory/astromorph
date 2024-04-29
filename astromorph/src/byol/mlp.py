from typing import Union, Type

import torch.distributed as distributed
from torch import nn


def parrallel_or_single_batchnorm() -> (
    Union[Type[nn.BatchNorm1d], Type[nn.SyncBatchNorm]]
):
    """Give a parallel or regular BatchNorm.

    Depending on the environment (parallellized training or not), you need
    a different batch normalization.

    Returns:
        a BatchNorm class
    """
    if distributed.is_initialized() and distributed.get_world_size() > 1:
        return nn.SyncBatchNorm
    else:
        return nn.BatchNorm1d


class MultiLayerPerceptron(nn.Module):
    """Standard multilayer perceptron for calculating the projections.

    Attributes:
        model: The actual perception
    """
    def __init__(self, representation_size: int, hidden_size: int, projection_size: int) -> None:
        """Init the MLP.

        Args:
            representation_size: size of input vector
            hidden_size: size of hidden layer
            projection_size: size of output vector
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(representation_size, hidden_size),
            parrallel_or_single_batchnorm()(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.model(x)
