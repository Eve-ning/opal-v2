from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# %%
from opal.model.delta_model import DeltaModel


def bezier_component(x, i, n):
    return comb(n, i) * ((1 - x) ** (n - i)) * (x**i)


m = DeltaModel.load_from_checkpoint(Path("epoch=9-step=3891.ckpt"))
m: DeltaModel
uid_i = 12
ws = m.emb_uid_curve.weight[uid_i].detach().numpy()


z = torch.tensor(
    [
        [bezier_component(t, e, len(ws) - 1) for e, w in enumerate(ws)]
        for t in np.linspace(0, 1, 100)
    ]
)


plt.plot((z * ws).sum(dim=1))
plt.title(f"uid_i={m.le_uid.classes_[uid_i]}")
plt.show()

# %%

m.le_mid.classes_[12]
# %%
import pandas as pd

df = pd.DataFrame(
    {"w": m.emb_mid.weight.squeeze().detach().numpy(), "id": m.le_mid.classes_}
)
# %%
