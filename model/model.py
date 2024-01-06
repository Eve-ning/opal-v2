from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import nn

from model.exp_linear import ExpLinear


class Model(pl.LightningModule):
    def __init__(
        self,
        uid_le: LabelEncoder,
        mid_le: LabelEncoder,
        acc_qt: QuantileTransformer,
        ln_ratio_weights: list,
        rc_emb: int = 1,
        ln_emb: int = 1,
        rc_delta_emb: int = 3,
        ln_delta_emb: int = 3,
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
        self.uid_emb_rc = nn.Embedding(n_uid, rc_emb)
        self.mid_emb_rc = nn.Embedding(n_mid, rc_emb)
        self.uid_emb_ln = nn.Embedding(n_uid, ln_emb)
        self.mid_emb_ln = nn.Embedding(n_mid, ln_emb)
        self.emb_rc_bn = nn.BatchNorm1d(rc_emb, affine=False)
        self.emb_ln_bn = nn.BatchNorm1d(ln_emb, affine=False)
        self.delta_rc_to_acc = nn.Sequential(
            ExpLinear(rc_emb, rc_delta_emb),
            nn.ReLU(),
            ExpLinear(rc_delta_emb, 1),
            nn.Sigmoid(),
        )
        self.delta_ln_to_acc = nn.Sequential(
            ExpLinear(ln_emb, ln_delta_emb),
            nn.ReLU(),
            ExpLinear(ln_delta_emb, 1),
            nn.Sigmoid(),
        )
        self.save_hyperparameters()

    def forward(self, x_uid, x_mid):
        w_ln_ratio = self.ln_ratio_weights[x_mid]
        x_uid_emb_rc = self.uid_emb_rc(x_uid)
        x_mid_emb_rc = self.mid_emb_rc(x_mid)
        x_uid_emb_ln = self.uid_emb_ln(x_uid)
        x_mid_emb_ln = self.mid_emb_ln(x_mid)

        x_rc_delta = (x_uid_emb_rc - x_mid_emb_rc) / 2
        x_ln_delta = (x_uid_emb_ln - x_mid_emb_ln) / 2

        x_rc_acc = self.delta_rc_to_acc(self.emb_rc_bn(x_rc_delta))
        x_ln_acc = self.delta_ln_to_acc(self.emb_ln_bn(x_ln_delta))

        y = (
            x_rc_acc.squeeze() * (1 - w_ln_ratio)
            + x_ln_acc.squeeze() * w_ln_ratio
        )
        return y

    def step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x_uid, x_mid, y = batch
        y_pred = self(x_uid, x_mid)
        loss = nn.MSELoss()(y_pred, y)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x_uid, x_mid, y = batch
        y_pred = self(x_uid, x_mid)
        loss = nn.MSELoss()(y_pred, y)
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
        return torch.optim.Adam(self.parameters(), lr=0.003, weight_decay=1e-4)

    def decode_acc(self, x):
        return self.acc_qt.inverse_transform(x.cpu().reshape(-1, 1))

    def encode_acc(self, x):
        return self.acc_qt.transform(x.cpu().reshape(-1, 1)).squeeze()

    def decode_uid(self, x):
        return self.uid_le.inverse_transform(x.cpu().reshape(-1, 1))

    def decode_mid(self, x):
        return self.mid_le.inverse_transform(x.cpu().reshape(-1, 1))

    def encode_uid(self, x):
        return self.uid_le.transform(x.cpu().reshape(-1, 1)).squeeze()

    def encode_mid(self, x):
        return self.mid_le.transform(x.cpu().reshape(-1, 1)).squeeze()
