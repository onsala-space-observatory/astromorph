from typing import Union

import numpy as np
import torch

from astropy.io import fits
from torch.utils.data import Dataset

from .helpers import augment_image, make_4D


class FilelistDataset(Dataset):
    """A class to gather multiple FITS images in a Dataset."""

    def __init__(self, filelist: Union[str, list], *args, **kwargs):
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
        image = self.read_fits_data(self.filenames[index])
        images = augment_image(image)

        return images

    def read_fits_data(self, filename: str):
        # FITS data is standard in dtype '>f4', convert to float before converting to tensor
        data = fits.getdata(filename).astype(float)
        return torch.from_numpy(data).float()

    def get_all_items(self):
        """Produce all items as inferable images

        Returns:
            list of 4D torch Tensors that can be used for inference
        """
        return [make_4D(self.read_fits_data(filename)) for filename in self.filenames]

    def get_object_property(self, keyword: str):
        """Retrieve an object property from the FITS header

        Args:
            keyword: property keyword in the FITS file header

        Returns:
            a FITS header property
        """
        object_properties = []
        for filename in self.filenames:
            header = fits.open(filename).pop().header
            try:
                object_property = header[keyword]
            except KeyError:
                object_property = "N/A"
            object_properties.append(object_property)
        return object_properties
