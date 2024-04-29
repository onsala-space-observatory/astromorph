from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, stacksize: int = 1) -> None:
        self.stacksize = stacksize

