import numpy as np
import torch

from astropy.io import fits
from scipy.ndimage import find_objects, label
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
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


def make_masked_image(
    data: np.ndarray, mask: np.ndarray, label: int = 1, median_fill: bool = True
):
    """Filter out the pixels in an image that are not part of the mask.

    Args:
        data: original data
        mask: labeled image mask
        label: which part of the mask to use
        median_fill: whether to fill the excised area with gaussian noise,
                     centered on the median

    Returns:
        a masked data array
    """
    median = np.median(data)
    std = data.std()

    condition = mask == label
    data *= condition
    if median_fill:
        noise = np.random.normal(loc=median, scale=std, size=data.shape)
        data[~condition] = noise[~condition]

    return data


class MaskedDataset(BaseDataset):
    def __init__(
        self,
        datafile: str,
        maskfile: str,
        remove_unrelated_data: bool = False,
        median_fill: bool = True,
        *args, **kwargs
    ):
        """Retrieve a list of arrays containing image data, based on the raw data and a mask.

        Args:
            datafile: filename for the real (raw) data
            maskfile: filename for the mask data
            remove_unrelated_data: if True, set all pixels outside a mask to 0
        """
        super().__init__(*args, **kwargs)
        # Read maskdata and real data into numpy array
        real_data = fits.getdata(datafile).astype(float)
        mask_data = fits.getdata(maskfile).astype(int)

        labels, n_features = label(mask_data)

        # We extract a list with the slices of all the objects
        # xy_slices has datatype List[Tuple[np.slice, np.slice]]
        xy_slices = find_objects(labels)
        threshold = 5
        large_object_slices = [
            (
                xy_slice,
                # offset by 1, because enumerate starts at 0, and labels at 1
                label + 1,
            )
            for label, xy_slice in enumerate(xy_slices)
            if (xy_slice[0].stop - xy_slice[0].start > threshold)
            and (xy_slice[1].stop - xy_slice[1].start > threshold)
        ]

        if remove_unrelated_data:
            cloud_images = [
                # Slice first, then filter by label.
                # That is faster than searching the entire image for the label value
                make_masked_image(
                    real_data[xy_slice], labels[xy_slice], label, median_fill
                )
                for xy_slice, label in large_object_slices
            ]
        else:
            cloud_images = [real_data[xy_slice] for xy_slice, _ in large_object_slices]

        self.objects = [torch.from_numpy(cloud_clipping(image)).float() for image in cloud_images]

    def __len__(self):
        """Return the size of the dataset.

        Returns:
            the number of objects in the dataset.
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

        return augment_image(image, stacksize=self.stacksize)


    def get_all_items(self):
        """Produce all items as inferable images

        Returns:
            list of 4D torch Tensors that can be used for inference
        """
        return [make_4D(image, stacksize=self.stacksize) for image in self.objects]
