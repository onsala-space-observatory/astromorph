import argparse
import os
import tomllib
from typing import Union

from datasets import MaskedDataset, FilelistDataset
from torchvision import models
import torch
from torch.utils.tensorboard import SummaryWriter
from byol_pytorch import BYOL
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from sklearn import cluster

# Provide these to the namespace for the read models
from basic_training import RandomApply, BYOL
from models import NLayerResnet
from settings import InferenceSettings


def pad_image_to_square(image):
    """Convert image to a square image.

    The image is padded with zeros where necessary

    Args:
        image (np.ndarray): an image of shape (channels, width, height)

    Returns: a square image of shape (channels, new_size, new_size)

    """
    delta = abs(image.shape[1] - image.shape[2])
    fixed_axis = np.argmax(image.shape)
    expand_axis = 1 + (1 - (fixed_axis - 1))
    d1 = delta // 2
    d2 = delta - d1

    pad_widths = [
        (0, 0),
        (d1, d2) if expand_axis == 1 else (0, 0),
        (d1, d2) if expand_axis == 2 else (0, 0),
    ]

    return np.pad(image, pad_width=pad_widths)


def normalize_image(image):
    """Ensure that an image has pixel values between 0 and 1

    Args:
        image (np.ndarray): an image to be normalized
    """
    image -= image.min()
    image /= image.max()
    return image


def main(dataset: Union[MaskedDataset, FilelistDataset], model_name: str):
    """Run the inference.

    Args:
        datafile: FITS filename with the data
        maskfile: FITS filename with the binary mask
        model_name: filename of the trained neural network
    """

    # images is a list of tensors with shape (1, 3, width, height)
    images = dataset.get_all_items()

    # Loading model
    print(f"Loading pretrained model {model_name}...")

    learner = torch.load(model_name)
    learner.eval()

    print("Calculating embeddings...")
    with torch.no_grad():
        _, dummy_embeddings = learner(torch.from_numpy(images[0]), return_embedding=True)
        embeddings_dim = dummy_embeddings.shape[1]
        embeddings = torch.empty((0, embeddings_dim))
        for image in tqdm(images):
            proj, emb = learner(torch.from_numpy(image), return_embedding=True)
            embeddings = torch.cat((embeddings, emb), dim=0)

    print("Clustering embeddings...")
    clusterer = cluster.KMeans(n_clusters=10)
    cluster_labels = clusterer.fit_predict(embeddings)

    print("Producing thumbnails...")
    plot_images = [normalize_image(image) for image in images]

    # If thumbnails are too large, TensorBoard runs out of memory
    thumbnail_size = 144
    resized = [
        # Use the [None] to remove the first extraneouos dimension
        resize(pad_image_to_square(im[0]), (3, thumbnail_size, thumbnail_size))[None]
        for im in plot_images
    ]

    # Concatenate thumbnails into a single tensor for labelling the embeddings
    all_ims = torch.cat([torch.from_numpy(ri) for ri in resized])

    # Remove directory names, and remove the extension as well
    model_basename = os.path.basename(model_name).split(".")[0]
    writer = SummaryWriter(log_dir=f"runs/{model_basename}/")

    # If the data is stored in FITS files, retrieve extra metadata
    if isinstance(dataset, FilelistDataset):
        # Retrieve object name, RA, dec, rest frequency, and the filename
        names = dataset.get_object_property("OBJECT")
        right_ascension = dataset.get_object_property("OBSRA")
        declination = dataset.get_object_property("OBSDEC")
        rest_freq = dataset.get_object_property("RESTFRQ")
        filenames = dataset.filenames
        labels = list(
            zip(
                cluster_labels,
                names,
                right_ascension,
                declination,
                rest_freq,
                filenames,
            )
        )
        writer.add_embedding(
            embeddings,
            label_img=all_ims,
            metadata=labels,
            metadata_header=[
                "cluster",
                "object",
                "right ascension",
                "declination",
                "rest freq",
                "filepath",
            ],
        )
    else:
        writer.add_embedding(embeddings, label_img=all_ims, metadata=cluster_labels)


if __name__ == "__main__":
    # Options can either be provided by command line arguments, or a config file
    # Options from the command line will override those from the config file
    parser = argparse.ArgumentParser(
        prog="Astromorph pipeline", description=None, epilog=None
    )
    parser.add_argument("-d", "--datafile", help="Define a data file")
    parser.add_argument("-m", "--maskfile", help="Specify a mask file")
    parser.add_argument("-n", "--trained_network_name", help="Saved network model")
    parser.add_argument("-c", "--configfile", help="Specify a config file")
    args = parser.parse_args()

    # If there is a config file, load those settings first
    # Otherwise, only use settings from the command line
    if args.configfile:
        overriding_settings = vars(args)
        configfile = overriding_settings.pop("configfile")
        with open(configfile, "rb") as file:
            config_dict = tomllib.load(file)
        # Overwrite the config file settings with command line settings
        for key, value in overriding_settings.items():
            if value is not None:
                config_dict.update({key: value})
    else:
        config_dict = vars(args)

    # Use InferenceSettings to validate settings
    settings = InferenceSettings(**config_dict)

    print("Reading data")
    if args.maskfile:
        dataset = MaskedDataset(settings.datafile, settings.maskfile)
    else:
        dataset = FilelistDataset(settings.datafile)

    main(dataset, settings.trained_network_name)
