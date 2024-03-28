from typing import Union

import torch

from astropy.io import fits
from torch.utils.data import Dataset

from .helpers import augment_image, make_4D


class FilelistDataset(Dataset):
    """A class to gather multiple FITS images in a Dataset."""

    def __init__(self, filelist: Union[str, list]):
        """Create a FilelistDataset.

        This will only store the filenames in memory.
        Files will be opened and loaded on an as-needed basis.

        Args:
            filelist: filename of the file containing all FITS filenames,
                      or a list of these filenames
        """
        if isinstance(filelist, list):
            self.filenames = filelist
        else:
            with open(filelist, "r") as file:
                # Make sure to remove the newline characters at the end of each filename
                self.filenames = [fname.strip("\n") for fname in file.readlines()]

    def __len__(self):
        """Return the size of the dataset.

        Returns:
            the number of objects in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, index: int):
        """Retrieve the item at index.

        This will first open the FITS file, and retrieve multiple versions 
        of the image:
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
        image = fits.open(name=self.filenames[index]).pop().data
        images = augment_image(image)

        return torch.from_numpy(images)

    def get_all_items(self):
        """Produce all items as inferable images

        Returns:
            list of 4D torch Tensors that can be used for inference
        """
        return [make_4D(fits.open(filename).pop().data) for filename in self.filenames]

    def get_object_names(self):
        object_names = []
        for filename in self.filenames:
            header = fits.open(filename).pop().header
            try:
                object_name = header["OBJECT"]
            except KeyError:
                object_name = "N/A"
            object_names.append(object_name)
        return object_names
