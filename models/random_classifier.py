import torch
import torch.nn as nn


class RandomClassifier(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        batch_size = x.size(0)

        return torch.rand(
            batch_size,
            10,
            device=x.device
        )