import argparse
from dataset import CloudDataset
from torchvision import models
import torch
from torch.utils.tensorboard import SummaryWriter
from byol_pytorch import BYOL
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from sklearn import cluster


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


def main(datafile: str, maskfile: str, model_name: str):
    """Run the inference.

    Args:
        datafile: FITS filename with the data
        maskfile: FITS filename with the binary mask
        model_name: filename of the trained neural network
    """
    print("Reading data...")
    ds = CloudDataset(
        datafile=datafile,
        maskfile=maskfile,
    )

    # images is a list of tensors with shape (1, 3, width, height)
    images = ds.get_all_items()

    # Loading model
    model = models.resnet18()
    print(f"Loading pretrained model {model_name}...")
    model.load_state_dict(torch.load(f"{model_name}"))

    learner = BYOL(model, image_size=256, hidden_layer="avgpool")
    learner.eval()

    print("Calculating embeddings...")
    # embeddings are of dimension 512
    dim_embeddings = 512
    embeddings = torch.empty((0, dim_embeddings))
    with torch.no_grad():
        for image in tqdm(images):
            proj, emb = learner(torch.from_numpy(image), return_embedding=True)
            embeddings = torch.cat((embeddings, emb), 0)

    print("Clustering embeddings...")
    clusterer = cluster.KMeans(n_clusters=10)
    cluster_labels = clusterer.fit_predict(embeddings)



    print("Producing thumbnails...")
    plot_images = [normalize_image(image) for image in images]
    
    # If thumbnails are too large, TensorBoard runs out of memory
    thumbnail_size = 150
    resized = [
        # Use the [None] to remove the first extraneouos dimension
        resize(pad_image_to_square(im[0]), (3, thumbnail_size, thumbnail_size))[None]
        for im in plot_images
    ]

    # Concatenate thumbnails into a single tensor for labelling the embeddings
    all_ims = torch.cat([torch.from_numpy(ri) for ri in resized])

    writer = SummaryWriter(log_dir="runs/test_projection_quad_labels/")
    writer.add_embedding(embeddings, label_img=all_ims, metadata=cluster_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Astromorph pipeline", description=None, epilog=None
    )
    parser.add_argument("-d", "--datafile", help="Define a data file", required=True)
    parser.add_argument("-m", "--maskfile", help="Specify a mask file", required=True)
    parser.add_argument("-n", "--network", help="Saved network model", required=True)
    args = parser.parse_args()
    main(args.datafile, args.maskfile, args.network)
