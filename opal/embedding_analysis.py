from pathlib import Path

import torch

from opal.data import df_k
from opal.model.delta_model import DeltaModel

# %%
PROJECT_DIR = Path(__file__).parent
m: DeltaModel = (
    DeltaModel.load_from_checkpoint(
        PROJECT_DIR
        # / "opal"
        / Path("opal/iyqa8fox/checkpoints/epoch=14-step=24975.ckpt")
    )
    .eval()
    .cpu()
)

# %%
m.emb_uid_rc.weight.shape
# %%
n_uid = len(m.le_uid.classes_)
n_mid = len(m.le_mid.classes_)
# %%

delta_rc = (
    m.emb_uid_rc.weight.unsqueeze(0) - m.emb_mid_rc.weight.unsqueeze(1)
) / 2
delta_rc = (delta_rc - m.bn_rc.running_mean) / torch.sqrt(
    m.bn_rc.running_var + 1e-5
)

# %%
delta_rc.min()
# %%
acc_rc_emb = m.delta_rc_to_acc(delta_rc.reshape(-1, 1)).reshape(n_mid, n_uid)
# %%
acc_rc_emb.min()
# %%
acc_rc = m.qt_acc.inverse_transform(
    acc_rc_emb.detach().numpy().reshape(-1, 1)
).reshape(n_mid, n_uid)
# %%
import pandas as pd

df = pd.DataFrame(acc_rc, index=m.le_mid.classes_, columns=m.le_uid.classes_)
# %%

acc_rc.max()

# %%
m(torch.tensor([0]), torch.tensor([0]))
