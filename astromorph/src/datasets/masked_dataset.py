import numpy as np
import torch

from astropy.io import fits
from scipy.ndimage import find_objects, label
from torch.utils.data import Dataset


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

        self.objects = cloud_images

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

        # Training the model requires multiple images in a single go, because
        # this is necessary for the projection in the BYOL architecture.
        # Since every image has a different size, we do this by creating
        # multiple copies of each image.
        # These copies follow the D2 = Z2 x Z2 symmetry group

        im_e = self.make_inferable(image)
        im_c = np.rot90(im_e, k=2, axes=(2, 3))
        im_b = np.flip(im_e, axis=(2, 3))
        im_bc = np.rot90(im_b, k=2, axes=(2, 3))

        # Concatenate along axis 0 to produce a tensor of shape (4, 3, W, H)
        images = np.concatenate(
            [
                im_e,
                im_c,
                im_b,
                im_bc,
            ],
            axis=0,
        )

        return torch.from_numpy(images)

    @classmethod
    def make_inferable(cls, image: np.ndarray):
        """Produce a version of the image that can be run on the inference network

        Args:
            image: 2D numpy array

        Returns:
            4D torch Tensor that can be used for inference
        """
        # Clip the most extreme values, and convert to logspace for better
        # detection of faint features
        image = np.log10(np.clip(image, a_min=1, a_max=100))
        # Create two extra dimensions
        image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
        # Create three channels per image (for RGB values)
        return np.concatenate([image, image, image], axis=1)

    def get_all_items(self):
        """Produce all items as inferable images

        Returns:
            list of 4D torch Tensors that can be used for inference
        """
        return [self.make_inferable(image) for image in self.objects]
