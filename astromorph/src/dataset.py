import numpy as np
import torch

from astropy.io import fits
from scipy.ndimage import find_objects, label
from torch.utils.data import Dataset


class CloudDataset(Dataset):
    def __init__(self, datafile: str, maskfile: str):
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

        # print("Looking for objects...")

        # t0 = perf_counter()
        labels, n_features = label(mask_data)
        # t1 = perf_counter()

        # print(f"Found {n_features} objects from mask in {(t1-t0):.3f} s")

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

        # print(f"Constructed {len(cloud_images)} images...")
        self.objects = cloud_images

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        image = self.objects[index]
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

        return torch.from_numpy(images)
