from math import comb

import torch


def bezier_factory(n):
    def bezier_component(t, i):
        """
               i         (n - 1)
        nCi * t * (1 - t)
        """
        return comb(n, i) * ((1 - t) ** (n - i)) * (t**i)

    return bezier_component


n_curve_emb = 5
n_batch = 16
x_emb = torch.rand([n_batch, n_curve_emb])
x_t = torch.rand([n_batch])
# %%
bezier_fn = bezier_factory(n_curve_emb - 1)
# %%
[
    bezier_fn(
        x_t,
    )
]
