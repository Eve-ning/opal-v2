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
m.uid_rc_emb.weight.shape
# %%
n_uid = len(m.uid_le.classes_)
n_mid = len(m.mid_le.classes_)
# %%

delta_rc = (
    m.uid_rc_emb.weight.unsqueeze(0) - m.mid_rc_emb.weight.unsqueeze(1)
) / 2
delta_rc = (delta_rc - m.rc_emb_bn.running_mean) / torch.sqrt(
    m.rc_emb_bn.running_var + 1e-5
)

# %%
delta_rc.min()
# %%
acc_rc_emb = m.delta_rc_to_acc(delta_rc.reshape(-1, 1)).reshape(n_mid, n_uid)
# %%
acc_rc_emb.min()
# %%
acc_rc = m.acc_qt.inverse_transform(
    acc_rc_emb.detach().numpy().reshape(-1, 1)
).reshape(n_mid, n_uid)
# %%
import pandas as pd

df = pd.DataFrame(acc_rc, index=m.mid_le.classes_, columns=m.uid_le.classes_)
# %%

acc_rc.max()

# %%
m(torch.tensor([0]), torch.tensor([0]))
