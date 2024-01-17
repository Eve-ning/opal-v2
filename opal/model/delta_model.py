from typing import Any

import numpy as np
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

    def forward(self, x_uid, x_mid):
        w_ln_ratio = self.ln_ratio_weights[x_mid]

        x_uid_rc_emb = self.uid_rc_emb(x_uid)
        x_mid_rc_emb = self.mid_rc_emb(x_mid)
        x_uid_ln_emb = self.uid_ln_emb(x_uid)
        x_mid_ln_emb = self.mid_ln_emb(x_mid)

        x_rc_emb_delta = x_uid_rc_emb - x_mid_rc_emb
        x_ln_emb_delta = x_uid_ln_emb - x_mid_ln_emb

        x_rc_emb_delta_bn = self.rc_emb_bn(x_rc_emb_delta)
        x_ln_emb_delta_bn = self.ln_emb_bn(x_ln_emb_delta)

        x_rc_acc = self.delta_rc_to_acc(x_rc_emb_delta_bn)
        x_ln_acc = self.delta_ln_to_acc(x_ln_emb_delta_bn)

        y = (
            x_rc_acc.squeeze() * (1 - w_ln_ratio)
            + x_ln_acc.squeeze() * w_ln_ratio
        )
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
            self.parameters(),
            lr=0.003,  # weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=2,
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
