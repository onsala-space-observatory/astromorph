from typing import Callable, Optional
import random

from loguru import logger
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from .byol import BYOL


class RandomApply(nn.Module):
    """A class to provide a probability-layer in a neural network.

    When added as a layer to a neural network, it has probability _p_ to apply
    function _fn_ to the input. In other cases it will just forward the input.

    Attributes:
        fn:
        p:
    """

    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class ByolTrainer(nn.Module):

    DEFAULT_AUGMENTATION_FUNCTION = nn.Sequential(
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=(0, 360)),
        RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
    )

    DEFAULT_OPTIMIZER = torch.optim.Adam

    def __init__(
        self,
        network: nn.Module,
        hidden_layer: str = "avgpool",
        representation_size: int = 128,
        augmentation_function: Optional[Callable] = None,
        optimizer: Optional[Callable] = None,
        learning_rate: float = 5.0e-6,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.augmentation_function = (
            augmentation_function
            if augmentation_function is not None
            else self.DEFAULT_AUGMENTATION_FUNCTION
        )

        self.byol = BYOL(
            network=network,
            hidden_layer=hidden_layer,
            augmentation_function=self.augmentation_function,
            representation_size=representation_size,
        )

        optimizer = self.DEFAULT_OPTIMIZER if optimizer is None else optimizer
        self.optimizer = optimizer(
            self.byol.parameters(), lr=learning_rate
        )

        self.to_device(device)

    def forward(self, x: torch.Tensor, return_errors: bool = False):
        """Run data through the model.

        The model will return either embeddings, or the errors.

        Args:
            x: input data
            return_errors: flag for returning errors instead of embeddings

        Returns:
            Embeddings or errors
        """
        return self.byol(x, return_errors=return_errors)

    def train_epoch(self, train_data: DataLoader, batch_size: int = 16):
        """Train the model for a single epoch.

        Args:
            train_data: the training data
            batch_size: batch size of the data

        Returns:
            the total loss
        """
        total_loss = 0.0
        batch_loss = None

        for i, image in enumerate(tqdm(train_data)):
            image = image[0].to(self.device)
            loss = self.byol(image, return_errors=True)

            batch_loss = batch_loss + loss if batch_loss else loss
            total_loss += loss.sum()

            if i % batch_size == 0 and i > 0:
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.byol.update_moving_average()
                batch_loss = None

        return total_loss

    def test(self, test_data: DataLoader):
        """Get out-of-sample errors on a test data set.

        Args:
            test_data: the test data set

        Returns:
            the out-of-sample test errors
        """
        loss = 0
        with torch.no_grad():
            self.byol.eval()
            for item in test_data:
                # The DataLoader will automatically wrap our data in an extra dimension
                item = item[0].to(self.device)
                ind_loss = self.byol(item, return_errors=True)
                loss += ind_loss.sum()
        return loss

    def train_model(
        self,
        train_data: DataLoader,
        test_data: DataLoader,
        epochs: int = 10,
        writer: Optional[SummaryWriter] = None,
        log_dir: str = "runs/",
        save_file: Optional[str] = None,
        **kwargs,
    ):

        """Train the BYOL model for a given number of epochs.

        Args:
            train_data: data for training
            test_data: data for evaluating out-of-sample performance
            epochs: number of epochs to train for
            writer:  
            log_dir: 
            save_file: 
            **kwargs: 
        """
        writer = SummaryWriter(log_dir=log_dir)
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_data, **kwargs)
            writer.add_scalar(
                "Train loss", train_loss / len(train_data), epoch, new_style=True
            )
            logger.info(
                f"[Epoch {epoch}] Training loss: {train_loss / len(train_data):.3e}"
            )

            test_loss = self.test(test_data)
            writer.add_scalar(
                "Test loss", test_loss / len(test_data), epoch, new_style=True
            )
            logger.info(
                f"[Epoch {epoch}] Test OOS loss: {test_loss / len(test_data):.3e}"
            )

        if save_file:
            torch.save(self, save_file)
            logger.info(f"Model saved to {save_file}")

    def to_device(self, device, *args, **kwargs):
            self.to(device, *args, **kwargs)
            self.device = device
