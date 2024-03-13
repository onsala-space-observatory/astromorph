import argparse
import datetime as dt
import random
from time import perf_counter

import torch
from astropy.io import fits
from byol_pytorch import BYOL
from scipy.ndimage import find_objects, label
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm

from dataset import CloudDataset


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
    learn_image = image.to(device)

    loss = learner(learn_image)
    return learner, loss


def train_epoch(learner, data, optimizer, device="cpu", writer=None, epoch=0):

    total_loss = 0
    batch_loss = None
    batch_size = 64

    epoch_length = len(data) // batch_size
    base_index = 10**(len(str(epoch_length))+1) * epoch

    for i, image in enumerate(tqdm(data)):
        image = image[0]
        learner, loss = train_single_image(learner, image, optimizer, device)
        batch_loss = batch_loss + loss if batch_loss else loss
        total_loss += loss.sum()
        if i % batch_size == 0 and i > 0:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if writer:
                writer.add_scalar("Batch loss", batch_loss.sum() / batch_size, base_index + (i // batch_size))
            batch_loss = None

    return learner, total_loss

def test_epoch(learner, test_data, device="cpu"):
    loss = 0 
    with torch.no_grad():
        learner.eval()
        for item in test_data:
            item = item[0]
            ind_loss = learner(item)
            loss += ind_loss.sum()
    return loss



def train(model, train_image_list, optimizer, epochs=10, device="cpu", test_image_list=None, timestamp=None):
    
    writer = SummaryWriter(log_dir=f"runs/{timestamp}/") if timestamp else SummaryWriter(log_dir=f"runs/")

    for epoch in range(epochs):
        model.train()
        model, loss = train_epoch(model, train_image_list, optimizer, device, writer=writer, epoch=epoch+1)
        writer.add_scalar("Train loss", loss / len(train_image_list), epoch, new_style=True)
        if test_image_list:
            test_loss = test_epoch(model, test_image_list, device=device)
            writer.add_scalar("Test loss", test_loss / len(test_image_list), epoch, new_style=True)
        torch.save(resnet.state_dict(), f"./improved_net_e_{epoch}_{epochs}_{start_time}.pt")

    return model


def main(datafile, maskfile, epochs):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = "cpu"
    resnet = models.resnet18().to(device)

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

    optimizer = torch.optim.Adam(learner.parameters(), lr=5e-6)
    all_objects = CloudDataset(datafile=datafile, maskfile=maskfile)

    rng = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(all_objects, [0.8, 0.2], generator=rng)

    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=True)
    start_time = dt.datetime.now().strftime("%Y%m%d_%H%M")

    model = train(learner, train_data, optimizer, epochs=epochs, device=device, test_image_list=test_data, timestamp=start_time)
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
