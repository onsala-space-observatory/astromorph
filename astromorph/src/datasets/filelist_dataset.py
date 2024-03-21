from typing import Union

import numpy as np
import torch

from astropy.io import fits
from torch.utils.data import Dataset

from .helpers import augment_image, make_4D


class FilelistDataset(Dataset):
    def __init__(self, filelist: Union[str, list]):
        if isinstance(filelist, list):
            self.filenames = filelist
        else:
            with open(filelist, "r") as file:
                self.filenames = [fname.strip("\n") for fname in file.readlines()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        image = fits.open(name=self.filenames[index]).pop().data
        images = augment_image(image)

        return torch.from_numpy(images)

    def get_all_items(self):
        return [make_4D(fits.open(filename).pop().data) for filename in self.filenames]
