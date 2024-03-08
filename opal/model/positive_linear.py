from torch import nn
from torch.nn.functional import softplus, linear


class PositiveLinear(nn.Linear):
    """Linear layer with positive weights

    Notes:
        The positive linear layer ensures the estimated function is
        monotonic increasing. This is done by simply hijacking the
        forward method of the Linear layer and applying a softplus
        activation to the weights.
    """

    def forward(self, x):
        return linear(x, softplus(self.weight), self.bias)
