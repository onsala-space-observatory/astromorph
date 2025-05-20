from pathlib import Path
from typing import Optional

import tomllib
import torch
import typer
from click import Context
from typer.core import TyperGroup

from astromorph import pipeline_01_training, pipeline_02_inference
from astromorph.datasets import FitsFilelistDataset
from astromorph.settings import InferenceSettings, TrainingSettings


class OrderCommands(TyperGroup):  # type: ignore
    def list_commands(self, ctx: Context) -> list[str]:
        return list(self.commands)


astromorph = typer.Typer(
    help="Astromorph tool",
    cls=OrderCommands,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


# Training command
@astromorph.command()
def training(
    configfile: Path = typer.Option(
        ..., "--configfile", "-c", help="Specify a configfile"
    ),
):
    """Run the training pipeline."""

    with open(configfile, "rb") as file:
        config_dict = tomllib.load(file)
    settings = TrainingSettings(**config_dict)

    if settings.core_limit:
        torch.set_num_threads(settings.core_limit)

    dataset = FitsFilelistDataset(settings.datafile, **(settings.data_settings))

    pipeline_01_training.main(dataset, settings=settings)


# Inference command
@astromorph.command()
def inference(
    datafile: Optional[Path] = typer.Option(
        None, "--datafile", "-d", help="Define a data file"
    ),
    maskfile: Optional[Path] = typer.Option(
        None, "--maskfile", "-m", help="Specify a mask file"
    ),
    trained_network_name: Optional[Path] = typer.Option(
        None, "--trained_network_name", "-n", help="Saved network model"
    ),
    configfile: Optional[Path] = typer.Option(
        None, "--configfile", "-c", help="Specify a config file"
    ),
):
    """Run the inference pipeline."""

    # If there is a config file, load those settings first
    if configfile:
        with open(configfile, "rb") as file:
            config_dict = tomllib.load(file)
    else:
        config_dict = {}

    # Override config with command line arguments
    if datafile:
        config_dict["datafile"] = str(datafile)
    if maskfile:
        config_dict["maskfile"] = str(maskfile)
    if trained_network_name:
        config_dict["trained_network_name"] = str(trained_network_name)

    settings = InferenceSettings(**config_dict)

    dataset = FitsFilelistDataset(settings.datafile, **(settings.data_settings))

    pipeline_02_inference.main(
        dataset, settings.trained_network_name, settings.export_to_csv
    )
