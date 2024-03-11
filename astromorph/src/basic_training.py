import argparse
import datetime as dt
import random
from time import perf_counter

import numpy as np
import torch
from astropy.io import fits
from byol_pytorch import BYOL
from scipy.ndimage import find_objects, label
from torch import nn
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def get_objects(datafile: str, maskfile: str):
    """Retrieve a list of arrays containing image data, based on the raw data and a mask.

    Args:
        datafile: filename for the real (raw) data
        maskfile: filename for the mask data
    """
    # Read maskdata and real data into numpy array
    real_data = fits.open(datafile).pop().data
    mask_data = fits.open(maskfile).pop().data
    # Reverse byteorder, because otherwise the scipy.ndimage.label cannot deal with it
    mask_data = mask_data.newbyteorder()

    print("Looking for objects...")

    t0 = perf_counter()
    labels, n_features = label(mask_data)
    t1 = perf_counter()

    print(f"Found {n_features} objects from mask in {(t1-t0):.3f} s")

    # We extract a list with the slices of all the objects
    # xy_slices has datatype List[Tuple[np.slice, np.slice]]
    xy_slices = find_objects(labels)
    threshold = 5
    large_object_slices = [
        xy_slice
        for xy_slice in xy_slices
        if (xy_slice[0].stop - xy_slice[0].start > threshold)
        and (xy_slice[1].stop - xy_slice[1].start > threshold)
    ]

    cloud_images = [real_data[xy_slice] for xy_slice in large_object_slices]

    print(f"Constructed {len(cloud_images)} images...")
    return cloud_images


def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)


def train_single_image(learner, image, optimizer, device="cpu"):
    # Model input has to satisfy two conditions:
    #   1) multiple images in one go (necessary for projection in BYOL)
    #   2) three channels per image (ResNet is trained on 3-channel images)
    # Create two extra axes
    im1 = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    # Create 3 channels by copying the image along axis 1
    im1 = np.concatenate([im1, im1, im1], axis=1)
    # Create a diagonally flipped copy of the image,
    im2 = np.flip(im1, axis=(2, 3))
    # Concatenate the new image along axis 0
    images = np.concatenate([im1, im2], axis=0)
    # images = torch.from_numpy(images).to(device)

    learn_image = torch.from_numpy(images).to(device)

    loss = learner(learn_image)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return learner, loss


def train_epoch(learner, data, optimizer, device="cpu"):

    total_loss = 0
    for image in tqdm(data):
        learner, loss = train_single_image(learner, image, optimizer, device)
        total_loss += loss.sum()

    return learner, total_loss


def train(model, image_list, optimizer, epochs=10, device="cpu"):
    for _ in range(epochs):
        model, loss = train_epoch(model, image_list, optimizer, device)

    return model


def main(datafile, maskfile, epochs):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = "cpu"
    resnet = models.resnet50().to(device)

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
        resnet,
        image_size=256,
        hidden_layer="avgpool",
        use_momentum=False,  # turn off momentum in the target encoder
        augment_fn=augmentation_function,
    )

    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
    images = get_objects(datafile, maskfile)
    start_time = dt.datetime.now().strftime("%Y%m%d_%H%M")

    model = train(learner, images, optimizer, epochs=epochs, device=device)
    torch.save(resnet.state_dict(), f"./improved_net_e_{epochs}_{start_time}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Astromorph pipeline", description=None, epilog=None
    )
    parser.add_argument("-d", "--datafile", help="Define a data file", required=True)
    parser.add_argument("-m", "--maskfile", help="Specify a mask file", required=True)
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=10, type=int)
    args = parser.parse_args()
    main(args.datafile, args.maskfile, args.epochs)
