import argparse
import datetime as dt
import os
import pprint
import tomllib

import torch
from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from byol import ByolTrainer, MinMaxNorm
from datasets import FilelistDataset
from models import DEFAULT_MODELS
from settings import TrainingSettings


def main(full_dataset: Dataset, settings: TrainingSettings):
    epochs = settings.epochs
    network_name = settings.network_name
    network_settings = settings.network_settings
    # Timestamp to identify training runs
    start_time = dt.datetime.now().strftime("%Y%m%d_%H%M")
    logger.add(f"logs/{start_time}.log")
    logger.info(
        "Starting training run with settings:\n{}",
        pprint.pformat(settings.model_dump()),
    )

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
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.3),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=(0, 360)),
        T.RandomApply([T.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2),
    )

    normalization_function = MinMaxNorm()

    lr_scheduler = ExponentialLR if settings.exponential_lr is True else None

    learner = ByolTrainer(
        network=network,
        hidden_layer="avgpool",
        augmentation_function=augmentation_function,
        **(settings.byol_settings),
        device=device,
        lr_scheduler=lr_scheduler,
        lr_scheduler_options={"gamma": settings.gamma},
        learning_rate=settings.learning_rate,
        normalization_function=normalization_function,
    )

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

    model_file_name = f"./saved_models/improved_net_e_{epochs}_{start_time}.pt"

    learner.train_model(
        train_data=train_data,
        test_data=test_data,
        epochs=epochs,
        log_dir=f"runs/{start_time}/",
        save_file=model_file_name,
        batch_size=settings.batch_size,
    )


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

    dataset = FilelistDataset(settings.datafile, **(settings.data_settings))

    main(dataset, settings=settings)
