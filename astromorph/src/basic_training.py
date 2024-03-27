import argparse
import datetime as dt
import random
from typing import Callable, Optional

import torch
from byol_pytorch import BYOL
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm

from datasets import MaskedDataset, FilelistDataset
from models import NLayerResnet


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


def train_single_image(learner, image, device="cpu"):
    learn_image = image.to(device)

    loss = learner(learn_image)
    return learner, loss


def train_epoch(
    learner: nn.Module,
    data: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
):
    """Train a model for a single epoch

    Args:
        learner: the model
        data: training data
        optimizer: optimizer used for fitting
        device: device on which the training calculations are run
        writer: SummaryWriter for TensorBoard logging
        epoch: the training epoch

    Returns:
        A tuple of updated learner, and the total training loss
    """
    # Set initial conditions
    total_loss = 0.0
    batch_loss = None  # batch_loss will be of type torch.nn.loss._Loss
    batch_size = 32  # 64

    # Define constants
    epoch_length = len(data) // batch_size
    # Separate epochs for intermediate logging
    base_index = 10 ** (len(str(epoch_length)) + 1) * epoch

    for i, image in enumerate(tqdm(data)):
        # The DataLoader will automatically wrap our data in an extra dimension
        image = image[0]

        # Forward pass
        learner, loss = train_single_image(learner, image, device)

        # Keep track of batch loss and epoch loss
        batch_loss = batch_loss + loss if batch_loss else loss
        total_loss += loss.sum()

        # Do backwards step after _batch_size_ iterations
        if i % batch_size == 0 and i > 0:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if writer:
                writer.add_scalar(
                    "Batch loss",
                    batch_loss.sum() / batch_size,
                    base_index + (i // batch_size),
                )
            batch_loss = None

    return learner, total_loss


def test_epoch(learner: nn.Module, test_data: DataLoader, device: str = "cpu"):
    """Calculate test loss on neural network.

    Args:
        learner: neural network
        test_data: out of sample data for testing
        device: device on which calculations take place

    Returns:

    """
    loss = 0
    # Use no_grad context manager to prevent keeping track of gradient
    with torch.no_grad():
        learner.eval()
        for item in test_data:
            # The DataLoader will automatically wrap our data in an extra dimension
            item = item[0]

            ind_loss = learner(item)
            loss += ind_loss.sum()
    return loss


def train(
    model: nn.Module,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    device: str = "cpu",
    test_data: Optional[DataLoader] = None,
    timestamp: Optional[str] = None,
    save_intermediates: bool = False,
):
    """Train a model

    Args:
        model: BYOL model to be trained
        train_data: training data
        optimizer: optimizer for finding the best weights and biases
        epochs: number of epochs to train for
        device: device on which to train
        test_data: data for out-of-sample validation
        timestamp: timestamp for logging purposes
        resnet: save this network component of the BYOL network after every epoch

    Returns:
        a trained model
    """
    # Create a writer for TensorBoard logging
    writer = (
        SummaryWriter(log_dir=f"runs/{timestamp}/")
        if timestamp
        else SummaryWriter(log_dir=f"runs/")
    )

    for epoch in range(epochs):
        # Ensure the model is set to training mode for gradient tracking
        model.train()
        model, loss = train_epoch(
            model, train_data, optimizer, device, writer=writer, epoch=epoch + 1
        )
        writer.add_scalar("Train loss", loss / len(train_data), epoch, new_style=True)

        # Out of sample testing
        if test_data:
            test_loss = test_epoch(model, test_data, device=device)
            writer.add_scalar(
                "Test loss", test_loss / len(test_data), epoch, new_style=True
            )
        # Save the network nested in the BYOL
        if save_intermediates is not None:
            torch.save(
                model,
                f"./saved_models/improved_net_e_{epoch}_{epochs}_{timestamp}.pt",
            )

    return model


def main(full_dataset: Dataset, epochs: int, last_layer: str = "layer4"):
    # Use a GPU if available
    # For now, we default to CPU learning, because the GPU memory overhead
    # makes GPU slower than CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = "cpu"

    # Load neural network and augmentation function, and combine into BYOL
    network = NLayerResnet(last_layer=last_layer).to(device)

    augmentation_function = torch.nn.Sequential(
        RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )

    learner = BYOL(
        network,
        image_size=256,
        hidden_layer="avgpool",
        use_momentum=False,  # turn off momentum in the target encoder
        augment_fn=augmentation_function,
    )

    # Create optimizer with the BYOL parameters
    optimizer = torch.optim.Adam(learner.parameters(), lr=5e-6)

    # Do train/test-split, and put into DataLoaders
    rng = torch.Generator().manual_seed(42)  # seeded RNG for reproducibility
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [0.8, 0.2], generator=rng
    )

    # DataLoaders have batch_size=1, because images have different sizes
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Timestamp to identify training runs
    start_time = dt.datetime.now().strftime("%Y%m%d_%H%M")

    model = train(
        learner,
        train_data,
        optimizer,
        epochs=epochs,
        device=device,
        test_data=test_data,
        timestamp=start_time,
        save_intermediates=True,
    )

    torch.save(model, f"./saved_models/improved_net_e_{epochs}_{start_time}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Astromorph pipeline", description=None, epilog=None
    )

    parser.add_argument("-d", "--datafile", help="Define a data file", required=True)
    parser.add_argument("-m", "--maskfile", help="Specify a mask file")
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=10, type=int)
    parser.add_argument(
        "-l",
        "--last-layer",
        help="Last convolutional ResNet layer",
        default="layer4",
        type=str,
    )
    args = parser.parse_args()

    if args.maskfile:
        dataset = MaskedDataset(args.datafile, args.maskfile)
    else:
        dataset = FilelistDataset(args.datafile)

    main(dataset, args.epochs, args.last_layer)
