from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import nn

from opal.model.positive_linear import PositiveLinear


class DeltaModel(pl.LightningModule):
    def __init__(
        self,
        uid_le: LabelEncoder,
        mid_le: LabelEncoder,
        acc_qt: QuantileTransformer,
        ln_ratio_weights: list,
        rc_emb: int = 1,
        ln_emb: int = 1,
    ):
        super().__init__()
        self.uid_le = uid_le
        self.mid_le = mid_le
        self.acc_qt = acc_qt
        self.ln_ratio_weights = nn.Parameter(
            torch.tensor(ln_ratio_weights, dtype=torch.float),
            requires_grad=False,
        )
        n_uid = len(uid_le.classes_)
        n_mid = len(mid_le.classes_)
        self.uid_rc_emb = nn.Embedding(n_uid, rc_emb)
        self.mid_rc_emb = nn.Embedding(n_mid, rc_emb)
        self.uid_ln_emb = nn.Embedding(n_uid, ln_emb)
        self.mid_ln_emb = nn.Embedding(n_mid, ln_emb)
        self.rc_emb_bn = nn.BatchNorm1d(rc_emb, affine=False)
        self.ln_emb_bn = nn.BatchNorm1d(ln_emb, affine=False)
        self.delta_rc_to_acc = nn.Sequential(
            PositiveLinear(rc_emb, 1),
            nn.Sigmoid(),
        )
        self.delta_ln_to_acc = nn.Sequential(
            PositiveLinear(ln_emb, 1),
            nn.Sigmoid(),
        )
        self.save_hyperparameters()

    def forward(self, uid, mid, mixup=None, shuf_idx=None):
        w_ln_ratio = self.ln_ratio_weights[mid]

        uid_rc_emb = self.uid_rc_emb(uid)
        mid_rc_emb = self.mid_rc_emb(mid)
        uid_ln_emb = self.uid_ln_emb(uid)
        mid_ln_emb = self.mid_ln_emb(mid)

        rc_emb_delta = uid_rc_emb - mid_rc_emb
        ln_emb_delta = uid_ln_emb - mid_ln_emb

        rc_emb = self.rc_emb_bn(rc_emb_delta)
        ln_emb = self.ln_emb_bn(ln_emb_delta)

        rc_acc = self.delta_rc_to_acc(rc_emb)
        ln_acc = self.delta_ln_to_acc(ln_emb)

        y = rc_acc.squeeze() * (1 - w_ln_ratio) + ln_acc.squeeze() * w_ln_ratio
        return y.squeeze()

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x_uid, x_mid, y = batch
        y_pred = self(x_uid, x_mid)
        loss = torch.sqrt(nn.MSELoss()(y_pred, y))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x_uid, x_mid, y = batch
        y_pred = self(x_uid, x_mid)
        loss = nn.MSELoss()(
            torch.tensor(self.decode_acc(y_pred)),
            torch.tensor(self.decode_acc(y)),
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x_uid, x_mid, y = batch
        y_pred = self(x_uid, x_mid)
        loss = nn.MSELoss()(
            torch.tensor(self.decode_acc(y_pred)),
            torch.tensor(self.decode_acc(y)),
        )
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.003  # , weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=1,
            factor=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def decode_acc(self, x):
        return self.acc_qt.inverse_transform(x.cpu().reshape(-1, 1))

    def encode_acc(self, x):
        return self.acc_qt.transform(x.cpu().reshape(-1, 1)).squeeze()

    def decode_uid(self, x):
        return self.uid_le.inverse_transform(x)

    def decode_mid(self, x):
        return self.mid_le.inverse_transform(x)

    def encode_uid(self, x):
        return self.uid_le.transform(x)

    def encode_mid(self, x):
        return self.mid_le.transform(x)

    @property
    def uid_classes(self) -> np.ndarray:
        return np.array(self.uid_le.classes_)

    @property
    def mid_classes(self) -> np.ndarray:
        return np.array(self.mid_le.classes_)

    def predict_all(self) -> pd.DataFrame:
        n_uid = len(self.uid_classes)
        n_mid = len(self.mid_classes)
        a = torch.cartesian_prod(torch.arange(n_uid), torch.arange(n_mid))
        uids = a[:, 0]
        mids = a[:, 1]
        accs = self.acc_qt.inverse_transform(
            self(uids, mids).detach().numpy().reshape(-1, 1)
        ).reshape(n_uid, n_mid)

        return pd.DataFrame(
            accs, columns=self.mid_classes, index=self.uid_classes
        )
