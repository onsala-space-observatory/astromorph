from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Class to contain attributes and methods for all astronomical datasets.

    Attributes:
        stacksize: How often an image should be repeated (stacked) to
                   accomodate pre-trained models.
    """

    def __init__(self, stacksize: int = 1) -> None:
        self.stacksize = stacksize
