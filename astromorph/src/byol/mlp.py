from typing import Union, Type

import torch.distributed as distributed
from torch import nn


def parrallel_or_single_batchnorm() -> (
    Union[Type[nn.BatchNorm1d], Type[nn.SyncBatchNorm]]
):
    if distributed.is_initialized() and distributed.get_world_size() > 1:
        return nn.SyncBatchNorm
    else:
        return nn.BatchNorm1d


class MultiLayerPerceptron(nn.Module):
    def __init__(self, representation_size, hidden_size, projection_size) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(representation_size, hidden_size),
            parrallel_or_single_batchnorm()(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.model(x)
