import torch.nn.functional as F
import torch
from torch import nn


class ExpLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        return F.linear(x, torch.exp(self.weight), self.bias)
