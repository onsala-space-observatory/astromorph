import numpy as np
import torch

from astropy.io import fits
from scipy.ndimage import find_objects, label
from torch.utils.data import Dataset

from .helpers import augment_image, make_4D 


def cloud_clipping(image: np.ndarray):
    """Clip extreme values, and convert to logspace.

    This helps with the detection of faint features.

    Args:
        image: original image

    Returns:
        an image with clipped pixel values
    """
    return np.log10(np.clip(image, a_min=1, a_max=100))


class MaskedDataset(Dataset):
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

        labels, n_features = label(mask_data)

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

        self.objects = [cloud_clipping(image) for image in cloud_images]

    def __len__(self):
        """Return the size of the dataset.

        Returns: the number of objects in the dataset.

        """
        return len(self.objects)

    def __getitem__(self, index: int):
        """Retrieve the item at index.

        This will retrieve multiple versions of the object:
            - original
            - rotated 180 degrees
            - flipped
            - flipped and rotated by 180 degrees.

        This is done for data augmentation.

        Args:
            index: which object to retrieve

        Returns:
            a 4D torch tensor

        """
        image = self.objects[index]

        images = augment_image(image)

        return torch.from_numpy(images)

    def get_all_items(self):
        """Produce all items as inferable images

        Returns:
            list of 4D torch Tensors that can be used for inference
        """
        return [make_4D(image) for image in self.objects]
