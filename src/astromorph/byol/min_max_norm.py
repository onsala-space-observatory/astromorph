import torch


class MinMaxNorm(torch.nn.Module):
    def forward(self, image: torch.Tensor):
        image -= image.min()
        image /= image.max()
        return image
