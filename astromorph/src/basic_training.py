import argparse
import datetime as dt
import os
import pprint
from typing import Optional

from loguru import logger
import tomllib
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from byol import BYOL
from datasets import MaskedDataset, FilelistDataset
from models import DEFAULT_MODELS
from settings import TrainingSettings


def train_single_image(learner, image, device="cpu"):
    learn_image = image.to(device)

    loss = learner(learn_image, return_errors=True)
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
    batch_size = 16  # 64

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
        batch_loss = batch_loss + loss if batch_loss is not None else loss
        total_loss += loss.sum()

        # Do backwards step after _batch_size_ iterations
        if i % batch_size == 0 and i > 0:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            learner.update_moving_average()
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
            item = item[0].to(device)

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

    learning_scheduler = ExponentialLR(optimizer, gamma=0.95)

    logger.debug("Using 1-based counting for epoch numbering")
    for epoch in range(1, epochs + 1):
        # Ensure the model is set to training mode for gradient tracking
        logger.info(f"[Epoch {epoch}] Learning rate: {learning_scheduler.get_lr()[0]:.3e}")
        model.train()
        model, loss = train_epoch(
            model, train_data, optimizer, device, writer=writer, epoch=epoch
        )
        writer.add_scalar("Train loss", loss / len(train_data), epoch, new_style=True)
        logger.info(f"[Epoch {epoch}] Training loss: {loss / len(train_data):.3e}")
        learning_scheduler.step()

        # Out of sample testing
        if test_data:
            test_loss = test_epoch(model, test_data, device=device)
            writer.add_scalar(
                "Test loss", test_loss / len(test_data), epoch, new_style=True
            )
            logger.info(f"[Epoch {epoch}] Test OOS loss: {test_loss / len(test_data):.3e}")

        # Save the network nested in the BYOL
        if save_intermediates is not None:
            save_file = f"./saved_models/improved_net_e_{epoch}_{epochs}_{timestamp}.pt"
            torch.save(model, save_file)
            logger.info(f"[Epoch {epoch}] Checkpoint saved to {save_file}")

    return model


def main(
    full_dataset: Dataset,
    epochs: int,
    network_name: str,
    network_settings: dict,
    settings: Optional[TrainingSettings] = None,
):
    # Timestamp to identify training runs
    start_time = dt.datetime.now().strftime("%Y%m%d_%H%M")
    logger.add(f"logs/{start_time}.log")
    if settings:
        logger.info("Starting training run with settings:\n{}", pprint.pformat(settings.model_dump()))

    # Use a GPU if available
    # For now, we default to CPU learning, because the GPU memory overhead
    # makes GPU slower than CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.debug("Using device {}", device)

    # Load neural network and augmentation function, and combine into BYOL
    network = DEFAULT_MODELS[network_name](**network_settings).to(device)

    augmentation_function = torch.nn.Sequential(
        T.RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=(0, 360)),
        T.RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )

    learner = BYOL(
        network,
        hidden_layer="avgpool",
        augment_fn=augmentation_function,
        **(settings.byol_settings)
    ).to(device)

    # Create optimizer with the BYOL parameters
    optimizer = torch.optim.Adam(learner.parameters(), lr=5e-6)

    # Do train/test-split, and put into DataLoaders
    rng = torch.Generator().manual_seed(42)  # seeded RNG for reproducibility
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [0.8, 0.2], generator=rng
    )

    # DataLoaders have batch_size=1, because images have different sizes
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)

    # If necessary, create the folder saved_models.
    # Also, ensure it does not show up in git
    savedir = "./saved_models"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        with open(f"{savedir}/.gitignore", "w") as file:
            lines = [".gitignore\n", "*.pt\n"]
            file.writelines(lines)

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

    model_file_name = f"./saved_models/improved_net_e_{epochs}_{start_time}.pt"
    torch.save(model, model_file_name)
    logger.info("Model saved to {}", model_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Astromorph pipeline", description=None, epilog=None
    )

    parser.add_argument(
        "-c", "--configfile", help="Specify a configfile", required=True
    )

    with open(parser.parse_args().configfile, "rb") as file:
        config_dict = tomllib.load(file)
    settings = TrainingSettings(**config_dict)

    if settings.core_limit:
        torch.set_num_threads(settings.core_limit)

    if settings.maskfile:
        dataset = MaskedDataset(
            settings.datafile, settings.maskfile, **(settings.data_settings)
        )
    else:
        dataset = FilelistDataset(settings.datafile, **(settings.data_settings))

    main(
        dataset,
        settings.epochs,
        settings.network_name,
        settings.network_settings,
        settings=settings,
    )
