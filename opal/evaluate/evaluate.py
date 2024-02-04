import pandas as pd
import torch
from pytorch_lightning import LightningModule

from opal.data import OsuDataModule


def prediction_error(m: LightningModule, dm: OsuDataModule) -> pd.DataFrame:
    x_uids, x_mids, y_accs = [], [], []
    for val in dm.val_dataloader():
        x_uids.append(val[0])
        x_mids.append(val[1])
        y_accs.append(val[2])

    x_uids = torch.cat(x_uids)
    x_mids = torch.cat(x_mids)
    x_uids_labels = dm.uid_le.inverse_transform(x_uids.detach().numpy())
    x_mids_labels = dm.mid_le.inverse_transform(x_mids.detach().numpy())
    y_accs = dm.acc_qt.inverse_transform(
        torch.cat(y_accs).detach().numpy().reshape(-1, 1)
    ).squeeze()
    y_pred = dm.acc_qt.inverse_transform(
        m(x_uids, x_mids).detach().numpy().reshape(-1, 1)
    ).squeeze()

    df = pd.DataFrame(
        {
            "uid": x_uids_labels,
            "mid": x_mids_labels,
            "y_true": y_accs,
            "y_pred": y_pred,
            "error": y_accs - y_pred,
        }
    )
    return df
