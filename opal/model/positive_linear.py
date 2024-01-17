import torch.nn.functional as F
import torch
from torch import nn


class PositiveLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.fn = torch.nn.Softplus()

    def forward(self, x):
        return F.linear(x, self.fn(self.weight), self.bias)
